import os
import pandas as pd

# def merge_xlsx_files(folder_path):

folder_paths = [os.path.join("datos","pumping","sqm","Bombeo sop"),os.path.join("datos","pumping","sqm","Bombeo mop")]
# merge_xlsx_files(folder_path)

merged_df = pd.DataFrame()

for folder_path in folder_paths:
# Loop through all .xlsx files in the folder
    for file in os.listdir(folder_path):
                # if file.endswith((".xls", ".xlsx")):
        file_path = os.path.join(folder_path, file)
        try:
            # Read the Excel file
            df = pd.read_excel(file_path, usecols=["Fecha final", "Flujo medio mensual"])#, engine=engine)

            # Convert "Fecha final" to datetime format (auto-detect format)
            df["Fecha final"] = pd.to_datetime(df["Fecha final"], errors="coerce")

            # Remove any rows with NaT in "Fecha final"
            df = df.dropna(subset=["Fecha final"])

            # Aggregate by month (mean of "Flujo medio mensual")
            df = df.set_index("Fecha final").resample("M").mean()

            # Rename the column to the file name (without extension)
            df.columns = [os.path.splitext(file)[0]]

            # Merge into the main DataFrame
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how='outer')

        except Exception as e:
            print(f"Error reading {file}: {e}")

# return merged_df
merged_df["Total"] = merged_df.sum(axis=1, skipna=True)
print(merged_df)
out_path = os.path.join("datos","pumping")
merged_df.to_csv(os.path.join(out_path, "SQM_pumping.csv"))

# Example usage


# Display the merged DataFrame
# import ace_tools as tools
# tools.display_dataframe_to_user(name="Merged Data", dataframe=merged_data)
