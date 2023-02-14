import streamlit as st
import pandas as pd

def main():
    st.title("Redcap CSV File Manipulator")

    uploaded_file = st.file_uploader("Upload a redcap CSV file", type="csv")
    if uploaded_file is not None:
        # Read the CSV file into a Pandas dataframe
        df = pd.read_csv(uploaded_file)

        # Do some data manipulation on the dataframe
        # For example, let's add a new column that calculates the sum of two existing columns
        # Identify duplicates based on a specific column, e.g., 'col1'
        duplicates = df[df.duplicated(subset=['participant_id'], keep=False)]

        # # Create a dictionary to map each duplicated value to the first non-NA value in other columns
        col_map = {}
        for col in df.columns:
            if col != 'participant_id':
                col_map[col] = duplicates.groupby('participant_id')[col].first()

        # # Replace NA values in the original DataFrame with values from the duplicate row
        for col, map_series in col_map.items():
            df.loc[df[col].isna(), col] = df['participant_id'].map(map_series)

        # # Drop the duplicate rows
        df.drop_duplicates(subset=['participant_id'], keep='first', inplace=True)
        df = df.drop(columns = ['redcap_record_metadata'])
        df['participant_id'] = df['participant_id'].astype(str)

        # Display the dataframe in the Streamlit app
        st.write(df)

        # Add a button to download the modified CSV file
        if st.button("Download modified CSV file"):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="modified_data.csv">Download modified CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
