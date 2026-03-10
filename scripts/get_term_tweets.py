"""
Extract tweets containing specific terms from JSON files.

This script processes JSON tweet data, extracting tweets that contain a specified
search term. Handles compressed (.gz) files by decompressing them first.
"""
import json
import re
import os
from tqdm import tqdm
import argparse
import subprocess


def process_file(fname, outfname, term):
    """
    Extract tweets containing a specific term from a JSON file.

    Searches for the term in tweet text (excluding @mentions) and appends
    matching tweets to output file.

    Args:
        fname: Input JSON file path (newline-delimited JSON)
        outfname: Output file path for matching tweets
        term: Search term to look for in tweets
    """
    outf = open(outfname, "a")
    jsonf = open(fname, "r")

    for line in jsonf.read().split("\n"):
        if len(line) == 0:
            continue

        j = json.loads(line)
        # Remove @mentions to avoid matching usernames
        no_user = re.sub(r'@\w+', '', j["text"])

        # Case-insensitive search for term
        if term.lower() in no_user.lower():
            outf.write("%s\n" % line)

    jsonf.close()
    outf.close()

def main(args):
    """
    Process multiple directories of tweet JSON files to extract term matches.

    Iterates through directories named 202209XX (September 2022 data),
    decompresses .gz files if needed, and extracts matching tweets.
    """
    for i in range(1, args.end + 1):
        # Format day number with leading zero if needed
        if i < 10:
            num = "0%d" % i
        else:
            num = str(i)

        dirname = "202209%s" % num

        # Process all files in directory
        for f in tqdm(os.listdir(dirname)):
            # Decompress .gz files
            if f[-3:] == ".gz":
                subprocess.run(["gunzip", os.path.join(dirname, f)])
                f = f[:-3]

            process_file(os.path.join(dirname, f), "%s_tweets.json" % args.term, args.term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, default=30, help="last json index to cycle through")
    parser.add_argument("--term", type=str, help="term to search for in tweets")
    args = parser.parse_args()
    main(args)
