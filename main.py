#Main entry point. Shows a menu so you can drive the whole project
#without remembering each script's name.

import os
import sys
import pandas as pd

import data
import train
import predict

def print_menu():
    print("")
    print("================================================")
    print("   LLM FINE-TUNER (DistilBERT + LoRA) -- Menu")
    print("================================================")
    print("  I.   Generate synthetic sentiment dataset")
    print("  II.  Train models (base + LoRA + full fine-tune)")
    print("  III. Classify a sentence (interactive)")
    print("  IV.  Show dataset summary")
    print("  V.   Show model comparison")
    print("  VI.  Launch API server (web demo)")
    print("  VII. Exit")
    print("------------------------------------------------")

def option_generate():
    print("\n>>> Generating synthetic sentiment dataset...")
    data.generate_dataset()

def option_train():
    print("\n>>> Starting training pipeline...")
    train.run_training()

def option_predict():
    try:
        predict.run_interactive()
    except FileNotFoundError as e:
        print("  [ERR] " + str(e))

def option_summary():
    if not os.path.exists(data.TRAIN_CSV):
        print("  [ERR] No dataset yet. Pick option I first.")
        return
    train_df, val_df, test_df = data.load_splits()
    print("\nDataset summary:")
    print("  train: " + str(len(train_df)) + " samples")
    print("  val:   " + str(len(val_df)) + " samples")
    print("  test:  " + str(len(test_df)) + " samples")
    print("  train label distribution:")
    print("    positive: " + str(int(train_df["label"].sum())))
    print("    negative: " + str(int(len(train_df) - train_df["label"].sum())))

def option_comparison():
    path = os.path.join("models", "comparison.csv")
    if not os.path.exists(path):
        print("  [ERR] No comparison yet. Pick option II first.")
        return
    comp = pd.read_csv(path)
    print("\nModel comparison (sorted by accuracy):")
    print(comp.sort_values("accuracy", ascending=False).to_string(index=False))
    report_path = os.path.join("models", "classification_report.txt")
    if os.path.exists(report_path):
        print("\nClassification report (LoRA model):")
        with open(report_path) as f:
            print(f.read())

def option_api():
    print("\n>>> Starting API server at http://localhost:8000")
    print("    Open that URL in your browser to use the web demo.")
    print("    Press Ctrl+C to stop the server.\n")
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)

def main():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip().upper()
        if choice == "I":
            option_generate()
        elif choice == "II":
            option_train()
        elif choice == "III":
            option_predict()
        elif choice == "IV":
            option_summary()
        elif choice == "V":
            option_comparison()
        elif choice == "VI":
            option_api()
        elif choice == "VII" or choice in ("Q", "EXIT", "QUIT"):
            print("Bye!")
            sys.exit(0)
        else:
            print("  [ERR] Unknown option. Try I, II, III, IV, V, VI or VII.")

if __name__ == "__main__":
    main()
