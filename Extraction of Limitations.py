import streamlit as st
import pymupdf as fitz
import os
import json
import re
import requests
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

# Disable Docker
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# --- Groq API Setup ---
api = os.getenv("GROQ_API_KEY") or "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"
headers = {"Authorization": f"Bearer {api}"}
try:
    r = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    print(f"🔍 Groq API Test: {r.status_code} - {r.text[:200]}")
except Exception as e:
    print(" Groq API not reachable:", e)

llm_config = LLMConfig(
    config_list=[{
        "model": "llama3-70b-8192",
        "api_key": api,
        "base_url": "https://api.groq.com/openai/v1",
        "max_tokens": 2000
    }],
    timeout=600,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={"use_docker": False}
)

limitation_agent = AssistantAgent(
    name="limitation_agent",
    llm_config=llm_config,
    system_message="""You are a research assistant specialized in identifying and generating limitations from research papers.

  Task 1: If a section titled “Limitations”, “Study Limitations”, or any close variant exists:
- Extract its **full content**, including all bullet points, sub-paragraphs, and lists.
- Do **not summarize** or truncate — preserve the **entire raw text** exactly as it appears.

  Task 2: If no such section exists:
- Analyze the full research paper thoroughly.
- **Generate a list of likely study limitations**, based on scope, methodology, assumptions, dataset, or other constraints.
- Keep the output **precise and well-formatted**.

  Strict Rules:
- Do **NOT include** “Future Work”, “Ethics”, “Acknowledgements”, or “References” unless they **explicitly mention limitations**.
- Return **only** the extracted or generated limitations — **no summaries, no extra explanations, no headings**.

   Difference between Limitation and Future Work:
- **Limitations** = Weaknesses or constraints in the current study (e.g., small dataset, biased sample, narrow scope).
- **Future Work** = Suggestions for what could be explored or improved in future studies (e.g., trying new models, extending to other domains).

"""
)

# --- File Readers ---
def load_pdf_text(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def read_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    paper_text = ''
    if 'abstractText' in data:
        paper_text += data['abstractText'] + '\n'
    if 'sections' in data:
        for section in data['sections']:
            heading = section.get('heading', '')
            text = section.get('text', '')
            paper_text += f"{heading}\n{text}\n"
    return paper_text

# --- Extraction Utilities ---
def extract_explicit_limitations_section(text):
    heading_pattern = re.compile(
        r'(?:\n|^)\s*(?:\d{0,2}[\.\)]?\s*)?(Limitations|Study Limitations|Limitations and Future Work)\s*\n',
        re.IGNORECASE
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return None
    start_idx = matches[0].end()
    next_heading_pattern = re.compile(
        r'(?:\n|^)\s*(?:\d+[\.\)]\s+)?((?!Limitations)(?:[A-Z][A-Z\s]{2,}|\d+\s+[A-Z][A-Za-z\s]*|FAQs|Ethics Statement))\n'
    )
    next_match = next_heading_pattern.search(text[start_idx:])
    end_idx = next_match.start() + start_idx if next_match else len(text)
    return text[start_idx:end_idx].strip()

def extract_limitation_paragraphs(text):
    keywords = [
        "limitation", "limitations", "drawback", "shortcoming",
        "bias", "confounding", "future work", "could be improved",
        "sample size", "not generalizable", "scalability", "small dataset"
    ]
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if any(k in p.lower() for k in keywords)]

def generate_candidate_limitations(paragraphs):
    combined = "\n\n".join(paragraphs)[:10000]
    if not combined.strip():
        return "No relevant paragraphs found to extract limitations."
    user_proxy.initiate_chat(
        limitation_agent,
        message=f"The following are limitation-related paragraphs from a research paper:\n\n{combined}\n\nPlease generate a paragraph explaining the limitations."
    )
    msg = limitation_agent.last_message()
    return msg["content"] if msg else " No response from LLM."

# --- PDF Modifier ---
def generate_pdf_without_limitations(original_path, limitations_text, output_path):
    doc = fitz.open(original_path)
    limitation_lines = [line.strip() for line in limitations_text.split("\n") if line.strip()]
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            b_text = block[4].strip()
            for lim_line in limitation_lines:
                if lim_line and lim_line in b_text:
                    page.add_redact_annot(block[:4], fill=(1, 1, 1))
                    break
        page.apply_redactions()
    doc.save(output_path)
    doc.close()

# --- Save to Central TXT File ---
def append_to_limitations_log(paper_name, clean_text):
    log_path = "all_limitations.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n========== {paper_name} ==========\n")
        f.write(clean_text.strip() + "\n")

# --- Streamlit App ---
st.set_page_config(page_title=" Research Paper Analyzer", layout="wide")
st.title(" Research Paper Analyzer")

file = st.file_uploader(" Upload PDF or JSON Research Paper", type=["pdf", "json"])

if file:
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    with st.spinner("Reading file..."):
        paper_text = load_pdf_text(temp_path) if file.name.endswith(".pdf") else read_json(temp_path)

    st.subheader(" Limitations")
    paper_name = file.name
    verbatim = extract_explicit_limitations_section(paper_text)

    if verbatim:
        st.success(" Found explicit Limitations section.")
        clean_verbatim = verbatim.strip()
        st.text_area("Limitations Extracted", clean_verbatim, height=250)
        st.download_button(" Download Limitations (.txt)", clean_verbatim, file_name="verbatim_limitations.txt")
        append_to_limitations_log(paper_name, clean_verbatim)

        if file.name.endswith(".pdf"):
            modified_pdf_path = f"modified_{file.name}"
            generate_pdf_without_limitations(temp_path, clean_verbatim, modified_pdf_path)
            with open(modified_pdf_path, "rb") as f:
                st.download_button(" Download PDF without Limitations", f, file_name=modified_pdf_path)

    else:
        st.warning(" No explicit section found. Searching for paragraphs...")
        paras = extract_limitation_paragraphs(paper_text)
        if paras:
            st.info(" Generating limitations using LLM...")
            draft = generate_candidate_limitations(paras)
            st.markdown("###  LLM-Generated Limitations")
            st.text_area("Output", draft, height=250)
            st.download_button(" Download LLM Limitations (.txt)", draft, file_name="llm_limitations.txt")
            append_to_limitations_log(paper_name, draft)

            if file.name.endswith(".pdf"):
                modified_pdf_path = f"modified_{file.name}"
                generate_pdf_without_limitations(temp_path, draft, modified_pdf_path)
                with open(modified_pdf_path, "rb") as f:
                    st.download_button(" Download PDF without Limitations", f, file_name=modified_pdf_path)
        else:
            st.info(" No clear paragraphs found. Using full-paper inference...")
            MAX_CHAR_LIMIT_FOR_GROQ = 18000
            truncated_text = paper_text[:MAX_CHAR_LIMIT_FOR_GROQ]
            if len(paper_text) > MAX_CHAR_LIMIT_FOR_GROQ:
                st.warning(" The paper was too long. Truncated content to fit Groq's token limit.")

            with st.spinner(" Asking LLM to infer limitations from the full paper..."):
                try:
                    user_proxy.initiate_chat(
                        limitation_agent,
                        message=f"This research paper may not have a clearly marked 'Limitations' section.\n\nPlease read the following full paper and infer the likely limitations:\n\n{truncated_text}"
                    )
                    full_msg = limitation_agent.last_message()
                    full_inferred = full_msg["content"] if full_msg else " No response from LLM."
                except Exception as e:
                    full_inferred = f" Error while contacting Groq API: {str(e)}"

            st.text_area(" Inferred Limitations", full_inferred, height=250)
            st.download_button(" Download Inferred (.txt)", full_inferred, file_name="inferred_limitations.txt")
            append_to_limitations_log(paper_name, full_inferred)

            if file.name.endswith(".pdf") and full_inferred.strip():
                modified_pdf_path = f"modified_{file.name}"
                generate_pdf_without_limitations(temp_path, full_inferred, modified_pdf_path)
                with open(modified_pdf_path, "rb") as f:
                    st.download_button(" Download PDF without Limitations", f, file_name=modified_pdf_path)

    os.remove(temp_path)

# Optionally show existing limitations file
if os.path.exists("all_limitations.txt"):
    with open("all_limitations.txt", "r", encoding="utf-8") as f:
        st.markdown("###  Collected Limitations Log")
        st.download_button(" Download All Limitations", f.read(), file_name="all_limitations.txt")
