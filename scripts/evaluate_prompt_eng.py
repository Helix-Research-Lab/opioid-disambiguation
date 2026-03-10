"""
Evaluate LLM-generated labels against manual annotations.

This script compares AI-generated classifications with human-annotated labels
to calculate performance metrics (accuracy, sensitivity, specificity).
Handles label cleaning and majority voting across multiple predictions.
"""
import pandas as pd
import argparse


def get_gpt_label(row, df):
    """
    Get the majority vote label for a tweet from multiple predictions.

    Args:
        row: DataFrame row containing tweet
        df: DataFrame with all predictions

    Returns:
        Most common label for the tweet, or "ERROR" if no labels found
    """
    subdf = df.loc[df["tweet"] == row["tweet"]]
    counts = subdf["gpt label"].value_counts()
    if len(counts) == 0:
        return "ERROR"
    return counts.index[0]

def clean_generated_label(row):
    """
    Normalize LLM-generated labels to standardized format.

    Maps various label formats (yes/Yes/indirectly -> Yes, no/No -> No,
    unsure/Unsure/uncertain -> Unsure) to consistent values.

    Args:
        row: DataFrame row containing "gpt label" column

    Returns:
        Standardized label string or "ERROR" for unrecognized labels
    """
    lab = row["gpt label"].strip()
    if lab == "yes" or lab == "Yes" or lab == "indirectly":
        return "Yes"
    elif lab == "no" or lab == "No":
        return "No"
    elif lab == "unsure" or lab == "Unsure" or lab == "uncertain":
        return "Unsure"
    else:
        return "ERROR"


def main(args):
    """
    Evaluate LLM predictions against manual labels.

    Calculates accuracy, sensitivity, and specificity by comparing
    LLM-generated labels with human annotations.
    """
    man = pd.read_csv(args.manual)
    df = pd.read_csv(args.gpt)

    # Clean and standardize labels
    print(df["gpt label"].unique())
    df["gpt label"] = df.apply(clean_generated_label, axis=1)
    print(df["gpt label"].unique())

    # Identify tweets with varying predictions across repetitions
    varying = []
    for tweet in df["tweet"].unique():
        subdf = df.loc[df["tweet"] == tweet]
        counts = subdf["gpt label"].value_counts()
        if len(counts) > 1:
            varying.append(tweet)
            print(tweet)
            print(counts)
            print("---")
    print("Number of tweets with varying predicted labels: %d" % len(varying))

    # Assign majority vote labels to manual dataset
    man["gpt label"] = man.apply(get_gpt_label, args=(df,), axis=1)
    print(man.head())

    # Report statistics
    unsure = man.loc[man["gpt label"] == "Unsure"]
    print("Number of tweets with unsure predicted label: %d" % len(unsure))

    pred_pos = man.loc[man["gpt label"] == "Yes"]
    pred_neg = man.loc[man["gpt label"] == "No"]

    print(man["gpt label"].unique())
    print(man.loc[man["gpt label"] == "ERROR"])
    # Calculate confusion matrix using specified manual label column
    tp = pred_pos.loc[pred_pos[args.label] == "Yes"]
    fp = pred_pos.loc[pred_pos[args.label] == "No"]
    tn = pred_neg.loc[pred_neg[args.label] == "No"]
    fn = pred_neg.loc[pred_neg[args.label] == "Yes"]

    # Calculate performance metrics
    acc = (len(tp) + len(tn)) / (len(pred_pos) + len(pred_neg))
    sens = len(tp) / (len(tp) + len(fn))
    spec = len(tn) / (len(tn) + len(fp))

    print("using %s label" % args.label)
    print("TP: %d, FP: %d, TN: %d, FN: %d" % (len(tp), len(fp), len(tn), len(fn)))
    print("Accuracy: %0.2f\nSensitivity: %0.2f\nSpecificity: %0.2f" % (acc, sens, spec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=str, default="general_100_drug_tweets_manual_labels.csv", help="manual labels for queried tweets")
    parser.add_argument("--gpt", type=str, help="gpt-generated labels with specified prompt setup")
    parser.add_argument("--label", type=str, help="manual label column to use for evaluation")
    args = parser.parse_args()
    main(args)
