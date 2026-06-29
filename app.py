import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.ranker import run_pipeline

st.set_page_config(
    page_title="Redrob Ranker Sandbox",
    layout="wide"
)

st.title("Redrob Candidate Discovery & Ranking Sandbox")
st.write(
    "Upload a candidate sample file (`.json` or `.jsonl`) "
    "to run the ranking pipeline end-to-end and download the ranked CSV output."
)

uploaded_file = st.file_uploader(
    "Choose a candidate file", 
    type=["json", "jsonl"],
    help="Upload candidates.json or similar format file"
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(uploaded_file.read())
        temp_in_path = Path(temp_in.name)

    # 2. Define temp output file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_out:
        temp_out_path = Path(temp_out.name)

    st.info("File uploaded successfully. Running ranking pipeline...")
    
    try:
        result = run_pipeline(
            candidates_path=temp_in_path,
            out_path=temp_out_path,
            config_path=ROOT / "config" / "jd_requirements.yaml",
            top_n=100
        )
        df = pd.read_csv(temp_out_path)
        
        st.success(f"Pipeline executed successfully in {result.elapsed_seconds:.2f} seconds!")
        st.write(f"Processed: **{result.candidates_processed}** | Excluded: **{result.candidates_excluded}**")
        st.subheader("Top Ranked Candidates Preview")
        st.dataframe(
            df,
            column_config={
                "candidate_id": "Candidate ID",
                "rank": "Rank",
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "reasoning": "Reasoning Justification"
            },
            hide_index=True,
            use_container_width=True
        )
        with open(temp_out_path, "r", encoding="utf-8") as f:
            csv_data = f.read()
            
        st.download_button(
            label="📥 Download Ranked CSV Submission",
            data=csv_data,
            file_name="ranked_submission.csv",
            mime="text/csv",
            help="Download the spec-compliant ranked CSV output."
        )
        
    except Exception as e:
        st.error(f"Error executing pipeline: {e}")
        
    finally:
        try:
            if temp_in_path.exists():
                temp_in_path.unlink()
            if temp_out_path.exists():
                temp_out_path.unlink()
        except:
            pass
