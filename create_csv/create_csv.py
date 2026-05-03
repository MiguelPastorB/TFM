import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import csv
import re
from datetime import datetime
LOGS_PER_FILE = 10000


def ingest_logs_to_csv_stream(gather_dir, output_csv):
    """
    Traverse the 'gather_dir', read .log files, extract the log message and its label 
    (using associated label files), and write each row directly to a CSV file.

    :param gather_dir: Directory containing the log files.
    :param output_csv: Output CSV file path.
    """

    # Open the CSV file in write mode (using "w" mode) and create a CSV DictWriter.
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["log_message", "source","label"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # Write column headers once.

        # Traverse the logs folder recursively
        for root, dirs, files in os.walk(gather_dir):
            for file in files:
                if ".log" in file:
                    file_path = os.path.join(root, file)
                    # Apply filtering conditions as used in your ingestion pipeline
                    if "attacker" in file_path:
                        continue
                    if "logs" not in file_path:
                        continue

                    # Retrieve labels from associated file (if it exists)
                    labels = get_labels(file_path)
                    with open(file_path, 'r', encoding='utf-8') as log_f:
                        # Retrieve the log source from path
                        #TO-DO
                        after_desktop = file_path.split("gather\\", 1)[-1]
                        source = after_desktop.split("\\", 1)[0]
                        
                        for line_number, line in enumerate(log_f, 1):
                            log = line.strip()
                            if not log or not re.search(r'[A-Za-z0-9]', log) :
                                continue
                            if line_number > LOGS_PER_FILE:
                                break
                            
                            # Get the label for the current line using your get_label function.
                            log_label = get_label(labels, line_number)
                            #timestamp=get_timestamp(line)
                            # Write the log and label immediately to CSV
                            writer.writerow({
                                #"timestamp" : timestamp,
                                "log_message": log,
                                "source" : source,
                                "label": log_label
                            })

    print(f"Logs have been ingested and written to '{output_csv}'.")


def get_timestamp(log):
    patrones = [
    (r"([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", "%b %d %H:%M:%S"),
    (r"\[(\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]", "%d/%b/%Y:%H:%M:%S %z"),
    (r'"timestamp"":\s*(\d+\.\d+)', "unix_float")
    ]
    for patron, formato in patrones:
        matches = re.findall(patron, log)
        if matches:
            match=matches[0]
            if formato == "unix_float":
                try:
                    return float(match)
                except ValueError:
                    print(f"No se pudo convertir a float: {match}")
            else:
                try:
                    print("MATCH",match)
                    dt_object = datetime.strptime(match, formato)
                    
                    if formato == "%b %d %H:%M:%S":
                        dt_object = dt_object.replace(year=2022)
                    print(dt_object)
                    return dt_object.timestamp() 
                    
                except ValueError:
                    print(f"No se pudo convertir '{match}' con el formato '{formato}'")

def read_labels(file_path):
    """
    Reads a log file and creates a dictionary mapping line numbers to labels.
    """
    line_to_labels = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line_number > LOGS_PER_FILE:
                break
            line = line.strip()
            if not line:
                continue
            # print("READING THE JSON")
            try:
                log_entry = json.loads(line)  # Parse JSON
                line_number = log_entry.get("line")
                # print("line_number", line_number)
                labels = log_entry.get("labels", [])
                #print(labels)
                label = ' | '.join(labels)
                # print(label)
                # rules = log_entry.get("rules", {})
                #label = labels[0]
                # rule = rules.get(main_label, [""])[0]
                # print("label", labels)
                if line_number is not None:
                    line_to_labels[line_number] = label  # Store in dictionary
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file_path}: {line}")

    return line_to_labels


def get_labels(file_path):
    # Change the file file_path
    # Substitute the gather word to the label word in the file_path variable
    file_path = file_path.replace("gather", "labels")
    # Get the label from the file file_path
    if (not (os.path.isfile(file_path))):
        return None
    else:
        return read_labels(file_path)


def get_json(file_path):
    file_path = file_path.replace("gather", "labels")
    # Get the label from the file file_path
    if (not (os.path.isfile(file_path))):
        return None
    return file_path


def get_label(labels, line_number):
    if labels == None:
        return "normal_log"
    elif ((label := labels.get(line_number)) == None):
        return "normal_log"
    else:
        return label

def split_dataset(file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Optionally, inspect the DataFrame
    print(df.head())
    print("Total samples:", len(df))
    #remove the empty timestamps
    #df['timestamp'].replace('', pd.NA, inplace=True)
    #df_with_timestamp = df.dropna(subset=['timestamp']).copy()
    #print("Samples With timestamps:", len(df_with_timestamp))
    # Split the DataFrame into train and evaluation sets
    # For example: 80% for training and 20% for evaluation
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    print("Types of labels", train_df['label'].unique())
    print("Types of source", train_df['source'].unique())
    print("Training samples:", len(train_df))
    print("Evaluation samples:", len(eval_df))


# Example usage:
if __name__ == "__main__":
    # Adjust path as needed.
    gather_directory = "C:\\Users\\elcrio\\Desktop\\TFG\\gather"
    output_csv_file = "C:\\Users\\elcrio\\Desktop\\TFG\\logs-dataset.csv"
    #ingest_logs_to_csv_stream(gather_directory, output_csv_file)
    split_dataset(output_csv_file)