import streamlit as st
import pandas as pd
import json
import io
import re
import datetime
import random
from itertools import combinations
from typing import Dict, List
import openai
from fuzzywuzzy import process
from textblob import TextBlob  # For spell checking


def load_abbreviations(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            st.write(df.head())  # Debugging: Show first few rows in Streamlit
            return df.applymap(lambda x: x.strip().upper() if isinstance(x, str) else x)
        except Exception as e:
            st.error(f"Error reading abbreviations file: {e}")
    return pd.DataFrame()


def load_class_words(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, encoding="utf-8")

def load_domain_rules() -> str:
    with open("domain_rules.txt", "r", encoding="utf-8") as f:
        return f.read()

def save_abbreviations(df: pd.DataFrame):
    df.to_csv("abbreviations.csv", index=False, encoding="utf-8")

def save_class_words(df: pd.DataFrame):
    df.to_csv("class_words.csv", index=False, encoding="utf-8")

def save_domain_rules(text: str):
    with open("domain_rules.txt", "w", encoding="utf-8") as f:
        f.write(text)

def load_data_dictionary(uploaded_file) -> pd.DataFrame:
    return pd.read_excel(uploaded_file)

def download_report(df):
    """Generate Excel file for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Validation Report", index=False)
    output.seek(0)
    return output

def check_column_parts(column_name: str, abbreviations: Dict[str, str], english_name: str, class_words: List[str]) -> List[str]:
    """
    Ensures that column names contain only approved abbreviations, class words, or terms from the English Name.
    Fix: Check both abbreviation keys, values, and the English Name.
    """
    column_parts = set(column_name.split('_'))  # Break down column name into parts
    english_terms = set(english_name.split())  # Convert English Name into a set

    # ✅ Fix: Include abbreviation keys, their values, class words, and English Name terms
    approved_terms = set(abbreviations.keys()).union(set(abbreviations.values())).union(set(class_words)).union(english_terms)

    # ✅ Keep all numeric values as valid
    def is_number(value):
        return bool(re.match(r"^\d+$", value))  # Check if value is purely numeric

    # Find missing parts
    missing_words = [part for part in column_parts if part not in approved_terms and not is_number(part)]

    return missing_words

def validate_class_word(column_name: str, data_type: str, english_name: str, class_word_type_map: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Validates if the class word (first three letters of the column name) matches the expected data type.
    If not, suggests a correction based on the provided English name.
    """
    class_word = column_name[:3]  # Extract first three letters
    expected_types = class_word_type_map.get(class_word, [])
    
    if data_type not in expected_types:
        # Determine replacement class word based on English name
        suggested_class_word = "TXT" if "text" in english_name.lower() else "IND"  # Default fallback
        
        return {
            "Validation Status": "FAIL",
            "Class Word Issue": f"Class word '{class_word}' does not match expected data type '{data_type}'.",
            "Suggested Class Word": suggested_class_word
        }
    
    return {"Validation Status": "PASS", "Class Word Issue": "Valid", "Suggested Class Word": ""}


def highlight_validation_status(val):
    if val == "FAIL":
        return 'background-color: orange; color: black; font-weight: bold;'
    return ''

def highlight_incorrect_capitalization(val):
    print(val, capitalize_english_name(val))
    if isinstance(val, str) and val != capitalize_english_name(val):
        return 'background-color: yellow; color: black; font-weight: bold;'
    return ''


def spell_check_description(description: str) -> dict:
    if not isinstance(description, str) or description.strip() == "":
        return {"Corrected Description": description, "Spelling Errors Found": "False"}

    blob = TextBlob(description)
    corrected_text = str(blob.correct())


    sentences = corrected_text.split(". ")
    corrected_sentences = []

    for sentence in sentences:
        words = sentence.split()
        if words:
            words[0] = words[0].capitalize()  # Capitalize first word
        corrected_sentences.append(" ".join(words))

    corrected_text = ". ".join(corrected_sentences)

    # Ensure "This" remains correctly capitalized if originally present
    if description.startswith("This ") and not corrected_text.startswith("This "):
        corrected_text = "This" + corrected_text[4:]

    # Fix incorrect word merging (e.g., "Thisis" → "This is")
    corrected_text = re.sub(r'(\bThis)([a-zA-Z])', r'\1 \2', corrected_text)  # Ensures "Thisis" → "This is"

    return {
        "Corrected Description": corrected_text,
        "Spelling Errors Found": "True" if corrected_text != description else "False"
    }

def capitalize_english_name(english_name) -> str:

    allowed_words = ["in","of","the","and","to","for","with","at","by","from","on","or"]
    
    if isinstance(english_name, dict):  # Extract 'Corrected Description' if it's a dictionary
        english_name = english_name.get("Corrected Description", "")

    if not isinstance(english_name, str):  # Ensure it's a string
        return str(english_name) if english_name is not None else ""

    words = english_name.split()
    capitalized_words = [word.capitalize() if i ==0 or word.lower() not in allowed_words else word.lower() for i, word in enumerate(words)]

    return " ".join(capitalized_words)

def call_openai_suggestion(
    table_name: str, column_name: str, english_name: str, 
    table_failure_reason: str, column_failure_reason: str, rules_text: str, abbreviations: Dict[str, str]
) -> Dict[str, str]:
    """
    Calls OpenAI API to suggest a corrected table and column name with additional validation notes.
    Ensures that non-abbreviation words remain unchanged.
    Uses abbreviations in the table name to correct potential truncations in the column name.
    """
    # Extract parts from table and column names
    table_parts = set(table_name.split('_'))
    column_parts = set(column_name.split('_'))

    # Find missing abbreviations in the column name
    missing_abbreviations = [part for part in table_parts if part in abbreviations and part not in column_parts]

    # Construct additional guidance for OpenAI
    abbreviation_guidance = (
        f"The table name '{table_name}' contains abbreviation(s) {', '.join(missing_abbreviations)}, "
        f"but the column name '{column_name}' does not include them. "
        "If it makes logical sense, restore missing abbreviations from the table name to the column name."
        if missing_abbreviations else "No missing abbreviations detected."
    )

    prompt = f"""
    You are an expert in database naming conventions.
    Based on the validation failure reasons, naming rules, and the given English Name, suggest a corrected table and column name.

    **Current Table Name**: {table_name}  
    **Current Column Name**: {column_name}  
    **English Name**: {english_name}  

    **Table Name Issue**:  
    {table_failure_reason}

    **Column Name Issue**:  
    {column_failure_reason}

    **Naming Rules**:  
    {rules_text}

    **Abbreviation Strategy**:  
    {abbreviation_guidance}

    Ensure that:
    - Non-abbreviation words remain unchanged.
    - If the table name contains recognized abbreviations that are missing from the column name, restore them if appropriate.
    - If a word is already in the approved abbreviation list, do not modify it.
    - Retain words like "HEADER" if they seem like a valid term rather than an abbreviation.

    Format your response as JSON:
    {{
      "Suggested Table Name": "NEW_TABLE_NAME",
      "Suggested Column Name": "NEW_COLUMN_NAME",
      "Additional Notes": "Explain any significant changes made."
    }}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "Suggested Table Name": table_name, 
            "Suggested Column Name": column_name, 
            "Additional Notes": "N/A", 
            "Error": str(e)
        }


def call_openai_for_sample_data(column_name: str, description: str, precision: int, scale: int, openai_api_key: str) -> List[str]:
    """
    Calls OpenAI API to generate sample data based on column name, description, precision, and scale.
    """
    openai.api_key = openai_api_key

    prompt = f"""
    Generate 3 realistic example values for a database column based on its name, description, precision, and scale.

    **Column Name**: {column_name}  
    **Description**: {description}  
    **Precision**: {precision}  
    **Scale**: {scale}  

    The sample data must align with the given precision and scale. Provide a JSON response in this format:
    {{
        "samples": ["example1", "example2", "example3"]
    }}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)["samples"]
    
    except Exception as e:
        return [f"Error: {str(e)}", "N/A", "N/A"]



def validate_abbreviation_usage(column_name: str, abbreviations: Dict[str, str], class_words: List[str], english_name: str) -> Dict[str, str]:
    """
    Validates abbreviations in the column name by checking them against the approved abbreviation list and class words.
    Ensures that no valid abbreviation or class word is incorrectly flagged as missing.
    """
    column_parts = set(column_name.split('_'))
    english_terms = set(english_name.split())  # Convert English Name into a set

    # Approved terms = abbreviations (keys & values) + class words + English name terms
    approved_terms = set(abbreviations.keys()).union(set(abbreviations.values())).union(set(class_words)).union(english_terms)

        # ✅ Keep all numeric values as valid
    def is_number(value):
        return bool(re.match(r"^\d+$", value))  # Check if value is purely numeric

    # Find missing parts
    unrecognized_parts = [part for part in column_parts if part not in approved_terms and not is_number(part)]

    # **Fix: If all words are valid, return a PASS state**
    if not unrecognized_parts:
        return {
            "unrecognized_parts": [],
            "missing_terms": [],
            "suggested_replacements": {}
        }

    return {
        "unrecognized_parts": unrecognized_parts,
        "missing_terms": [],
        "suggested_replacements": {}
    }
def call_openai_suggestion(
    table_name: str, column_name: str, english_name: str, 
    table_failure_reason: str, column_failure_reason: str, rules_text: str
) -> Dict[str, str]:
    """
    Calls OpenAI API to suggest a corrected table and column name with additional validation notes.
    Ensures that non-abbreviation words remain unchanged.
    """
    prompt = f"""
    You are an expert in database naming conventions.
    Based on the validation failure reasons and naming rules, suggest a corrected table and column name.

    **Current Table Name**: {table_name}  
    **Current Column Name**: {column_name}  
    **English Name**: {english_name}  

    **Table Name Issue**:  
    {table_failure_reason}

    **Column Name Issue**:  
    {column_failure_reason}

    **Naming Rules**:  
    {rules_text}

    Ensure that:
    - Non-abbreviation words remain unchanged.
    - If the table name does not follow the format T_*_*_*_FACT/DIM/STG/RPTNG, suggest a reasonable correction.
    - If the column name has abbreviations not in the approved list, correct them while preserving meaning.

    Format your response as JSON:
    {{
      "Suggested Table Name": "NEW_TABLE_NAME",
      "Suggested Column Name": "NEW_COLUMN_NAME",
      "Additional Notes": "Explain any significant changes made."
    }}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "Suggested Table Name": table_name, 
            "Suggested Column Name": column_name, 
            "Additional Notes": "N/A", 
            "Error": str(e)
        }


def generate_sample_data(data_type: str, precision: int, scale: int, column_name: str, description: str, openai_api_key: str) -> List[str]:
    """
    Generates three meaningful sample values based on column name, description, precision, and scale.
    Uses OpenAI API for ambiguous cases.
    """
    column_name_lower = column_name.lower()
    description_lower = description.lower() if description else ""
    samples = []

    if "date" in column_name_lower or "dob" in column_name_lower or "birth" in column_name_lower:
        # Generate realistic past dates
        for _ in range(3):
            samples.append((datetime.date.today() - datetime.timedelta(days=random.randint(5000, 30000))).strftime("%Y-%m-%d"))

    elif "id" in column_name_lower or "code" in column_name_lower or "ref" in column_name_lower:
        # Generate formatted IDs
        for _ in range(3):
            samples.append(f"ID{random.randint(1000, 9999)}")

    elif "price" in column_name_lower or "amount" in column_name_lower or "cost" in column_name_lower:
        # Generate realistic monetary values
        for _ in range(3):
            samples.append(f"${random.uniform(10, 10000):.{scale}f}")

    elif data_type.startswith("DECIMAL") or data_type.startswith("FLOAT"):
        # Generate realistic decimal values
        for _ in range(3):
            samples.append(f"{random.uniform(10**(precision-scale-1), 10**(precision-scale)):.{scale}f}")

    elif data_type.startswith("INT") or data_type.startswith("BIGINT"):
        # Generate realistic integer values
        for _ in range(3):
            samples.append(str(random.randint(10**(precision-1) if precision else 1, 10**precision-1 if precision else 9999)))

    else:
        # ✅ Pass precision & scale to OpenAI API for more accurate sample data
        samples = call_openai_for_sample_data(column_name, description, precision, scale, openai_api_key)

    return samples if samples else ["N/A", "N/A", "N/A"]  # Default case



def validate_data_dictionary(df: pd.DataFrame, class_words: List[str], abbreviations: Dict[str, str], openai_api_key: str) -> pd.DataFrame:
    """
    Validates the data dictionary, checks for class word mismatches, and generates sample data.
    """
    results = []
    
    class_word_type_map = {
        "TXT": ["VARCHAR","VARCHAR2", "TEXT"],
        "NAM": ["VARCHAR","VARCHAR2", "TEXT"],
        "CDE": ["INT", "BIGINT", "VARCHAR","VARCHAR2", "NUMBER"],
        "DTE": ["DATE", "DATETIME", "TIMESTAMP"],
        "TME": ["TIME", "DATETIME", "TIMESTAMP"],
        "IDN": ["INT", "BIGINT", "NUMBER"],
        "NBR": ["INT", "BIGINT",'NUMBER'],
        "AMT": ["DECIMAL", "FLOAT", "NUMERIC","NUMBER"],
        "CNT": ["INT", "BIGINT","NUMBER"],
        "IND": ["BOOLEAN", "CHAR", "VARCHAR", "VARCHAR2"]
    }
    
    for _, row in df.iterrows():
        table_name = str(row.get('Table Name', '')).strip().upper()
        column_name = str(row.get('Column Name', '')).strip().upper()
        english_name = str(row.get('English Name', '')).strip().upper()
        english_name1 = str(row.get('English Name', '')).strip()
        data_type = str(row.get('Data Type', '')).strip().upper()
        precision = int(row.get('Precision', 0)) if pd.notna(row.get('Precision')) else 0
        scale = int(row.get('Scale', 0)) if pd.notna(row.get('Scale')) else 0
        description = str(row.get('Description/Business Rules', '')).strip()

        #spell_check_result = spell_check_description(description)

        failure_reasons = []
        table_failure_reason = ""

        # Validate Table Name
        table_valid = table_name.startswith('T') and (
            table_name.endswith('FACT') or 
            table_name.endswith('DIM') or 
            table_name.endswith('STG') or 
            table_name.endswith('RPTNG')
        )
        if not table_valid:
            table_failure_reason = "Table name must start with 'T' and end with {'FACT','DIM','STG','RPTNG'}."

        # Validate Column Name Parts
        missing_parts = check_column_parts(column_name, abbreviations, english_name, class_words)

        # Validate Abbreviation Usage
        issues = validate_abbreviation_usage(column_name, abbreviations, class_words, english_name)

        # Validate Class Word Usage
        class_word_validation = validate_class_word(column_name, data_type, english_name, class_word_type_map)

        if class_word_validation["Validation Status"] == "FAIL":
            failure_reasons.append(class_word_validation["Class Word Issue"])

        # Remove "missing term" if already flagged as an "unrecognized abbreviation"
        filtered_missing_parts = [part for part in missing_parts if part not in issues["unrecognized_parts"]]

        if filtered_missing_parts:
            failure_reasons.append(f"Column name is missing or incorrect for: {', '.join(filtered_missing_parts)}")

        if issues["unrecognized_parts"]:
            failure_reasons.append(f"Unrecognized abbreviation(s): {', '.join(issues['unrecognized_parts'])} are not in the approved list.")

        column_failure_reason = "; ".join(failure_reasons)
        validation_status = "PASS" if not failure_reasons and not table_failure_reason else "FAIL"

        sample_data_records = generate_sample_data(data_type, precision, scale, column_name, description, openai_api_key)

        if validation_status == "FAIL":
            openai_suggestion = call_openai_suggestion(
                table_name, column_name, english_name, table_failure_reason, 
                column_failure_reason, load_domain_rules()
            )
            suggested_table_name = openai_suggestion.get("Suggested Table Name", table_name)
            suggested_column_name = openai_suggestion.get("Suggested Column Name", column_name)
            additional_notes = openai_suggestion.get("Additional Notes", "N/A")
        else:
            suggested_table_name = ""
            suggested_column_name = ""
            additional_notes = "No corrections needed."

        #english_name = spell_check_description(english_name)
        #english_name = capitalize_english_name(english_name)

        results.append({
            "Table Name": table_name,
            "Column Name": column_name,
            "English Name": english_name1,
            "Data Type": data_type,
            "Precision": precision,
            "Scale": scale,
            "Validation Status": validation_status,
            "Notes": "Valid" if validation_status == "PASS" else column_failure_reason or table_failure_reason,
            "Suggested Table Name": suggested_table_name,
            "Suggested Column Name": suggested_column_name,
            "Suggested Class Word": class_word_validation["Suggested Class Word"],
            "Additional Notes": additional_notes,
            "Corrected Description": description,
            "Sample Data 1": sample_data_records[0],
            "Sample Data 2": sample_data_records[1],
            "Sample Data 3": sample_data_records[2]
        })
    
    return pd.DataFrame(results)





def main():
    st.set_page_config(page_title="Data Dictionary Validator", layout="wide")
    st.title("📊 Data Dictionary Validator")
    openai_api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
    uploaded_dict = st.sidebar.file_uploader("📂 Upload Data Dictionary (Excel)", type=["xlsx"])
    uploaded_abb = st.sidebar.file_uploader("🔍 Upload Abbreviations (csv)", type=["csv"])
    uploaded_cw = st.sidebar.file_uploader("🔍 Upload Class Words (csv)", type=["csv"])
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Validation", "📖 Domain Rules", "🔍 Abbreviations", "🔍 Class Words"])
    
    with tab1:
        st.write("Please upload a Data Dictionary file to start the validation process.")
        
        if uploaded_dict and uploaded_abb and uploaded_cw:
            df_dict = load_data_dictionary(uploaded_dict)
            st.success("✅ All files uploaded successfully!")
            
            if st.button("🔍 Validate & Suggest Corrections"):
                with st.spinner("Processing validation..."):
                    abbreviations = load_abbreviations(uploaded_abb)
                    class_words = load_class_words(uploaded_cw)
                    domain_rules = load_domain_rules()
                    abbreviations_dict = abbreviations.set_index("NAME")["ABBR"].to_dict()
                    class_words_list = class_words["CLASS WORD"].tolist()
                    results_df = validate_data_dictionary(df_dict, class_words_list, abbreviations_dict, openai_api_key)
                    st.session_state["results_df"] = results_df
        
        if "results_df" in st.session_state:
            df_results = st.session_state["results_df"]
            
            if not df_results.empty:
                st.subheader("Validation Results")
                
                # Style dataframe
                styled_df = (
                    df_results.style
                    .applymap(highlight_validation_status, subset=["Validation Status"])
                    .applymap(highlight_incorrect_capitalization, subset=["English Name"])
                )


                # Display DataFrame
                st.dataframe(styled_df)

                # ✅ Dropdown for selecting failed records
                failed_records = df_results[df_results["Validation Status"] == "FAIL"]

                if not failed_records.empty:
                    # Generate list of failed Table-Column names
                    failed_options = failed_records.apply(lambda row: f"{row['Table Name']} - {row['Column Name']}", axis=1).tolist()
                    
                    # Auto-select the first failed record by default
                    selected_record = st.selectbox("🔍 Select a Failed Table-Column to View Details:", failed_options, index=0)

                    if selected_record:
                        # Get the selected row details
                        selected_row = failed_records[failed_records.apply(lambda row: f"{row['Table Name']} - {row['Column Name']}" == selected_record, axis=1)].iloc[0]

                        # Display details using st.write() instead of st.dataframe()
                        st.write("### **Validation Details**")
                        st.write(f"**Table Name:** {selected_row['Table Name']}")
                        st.write(f"**Column Name:** {selected_row['Column Name']}")
                        st.write(f"**Validation Status:** {selected_row['Validation Status']}")
                        st.write(f"**Notes:** {selected_row['Notes']}")
                        st.write(f"**Suggested Table Name:** {selected_row['Suggested Table Name']}")
                        st.write(f"**Suggested Column Name:** {selected_row['Suggested Column Name']}")
                        st.write(f"**Additional Notes:** {selected_row['Additional Notes']}")


                # ✅ Download Validation Report
                excel_file = download_report(df_results)
                st.download_button(
                    label="📥 Download Validation Report as Excel",
                    data=excel_file,
                    file_name="Validation_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with tab2:
        domain_rules = load_domain_rules()
        rules_text = st.text_area("Edit Domain Rules", domain_rules, height=300)
        if st.button("Save Domain Rules"):
            save_domain_rules(rules_text)
            st.success("Domain rules updated successfully!")
    
    with tab3:
        if uploaded_abb:
            abbreviations = load_abbreviations(uploaded_abb)
            abbreviations_text = st.data_editor(abbreviations, use_container_width=True)

    with tab4:
        if uploaded_cw:
            class_words = load_class_words(uploaded_cw)
            class_words_text = st.data_editor(class_words, use_container_width=True)

if __name__ == "__main__":
    main()
