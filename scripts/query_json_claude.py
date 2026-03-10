"""
Query tweets using Anthropic Claude API to classify opioid-related content.

This script supports both batch and individual query modes. The batch mode creates
batch API requests for processing multiple tweets efficiently. Individual mode
processes tweets in real-time with iterative prompting.
"""
import os
from dotenv import load_dotenv
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import pandas as pd
import argparse
import time
from tqdm import tqdm
import json


def batch_query(client, tweets, n):
    """
    Create a batch query request for Claude API.

    Args:
        client: Anthropic API client
        tweets: List of tweet batches to process
        n: Number of repetitions for each tweet batch
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."

    batch_queries = []
    for query_idx, query_tweets in enumerate(tweets):
        prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"

        for t in query_tweets:
            prompt += "<tweet>%s</tweet>\n" % t
        for repeat in range(n):
            r = Request(custom_id="tweet_query%d-%d" % (query_idx,repeat),
                        params=MessageCreateParamsNonStreaming(
                            model="claude-sonnet-4-5",
                            max_tokens=1000,
                            system=context,
                            messages=[{"role": "user",
                                       "content": prompt}]
                            )
                        )
            batch_queries.append(r)
    message_batch = client.messages.batches.create(requests=batch_queries)
    print(message_batch)


def batch_query_followup(client, tweets, responses):
    """
    Create followup batch queries to extract structured answers from reasoning.

    Args:
        client: Anthropic API client
        tweets: List of tweet batches
        responses: List of JSON responses from initial batch query
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."
    prompt2 = "Based on your reasoning above, answer the question in one word by saying \"yes\",\"no\", or \"unsure\" once for each tweet, where \"yes\" means that the tweet refers to opioids. Separate your answers by commas. Only give this in your response; do not add other content."

    # Parse responses and organize by query index
    response_dict = {}
    for r in responses:
        j = json.loads(r)
        query_id = j["custom_id"]
        query_idx = int(query_id.split("-")[0][11:])
        if not query_idx in response_dict.keys():
            response_dict[query_idx] = []
        if j["result"]["type"] == "succeeded":
            response_dict[query_idx].append(j["result"]["message"]["content"][0]["text"])
        else:
            response_dict[query_idx].append("APIError")

    batch_queries = []
    for query_idx, query_tweets in enumerate(tweets):
        prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"

        for t in query_tweets:
            prompt += "<tweet>%s</tweet>\n" % t

        for repeat, r in enumerate(response_dict[query_idx]):
            if r == "APIError":
                continue
            req = Request(custom_id="followup_query%d-%d" % (query_idx, repeat),
                          params=MessageCreateParamsNonStreaming(
                            model = "claude-sonnet-4-5",
                            max_tokens=24,
                            system=context,
                            messages=[{"role": "user",
                                       "content": prompt},
                                      {"role": "assistant",
                                       "content": r},
                                      {"role": "user",
                                       "content": prompt2}]
                            )
                          )
            batch_queries.append(req)
    message_batch = client.messages.batches.create(requests=batch_queries)
    print(message_batch)
        

def batch_to_labels(tweets, followup_responses, outname):
    """
    Convert batch API responses to labeled CSV output.

    Args:
        tweets: List of tweet batches
        followup_responses: List of JSON responses from followup batch query
        outname: Output CSV file path
    """
    if os.path.isfile(outname):
        og_df = pd.read_csv(outname)
    else:
        og_df = pd.DataFrame({"tweet": [], "claude label": []})
    tweetlist = []
    labellist = []
    for response in followup_responses:
        j = json.loads(response)
        query_id = j["custom_id"]
        query_idx = int(query_id.split("-")[0][14:])
        query_tweets = tweets[query_idx]
        labels = j["result"]["message"]["content"][0]["text"]
        assert len(query_tweets) == len(labels.split(","))
        for i in range(len(query_tweets)):
            tweetlist.append(query_tweets[i])
            labellist.append(labels.split(",")[i].strip())
    df = pd.DataFrame({"tweet": tweetlist, "claude label": labellist})
    updated_df = pd.concat([og_df, df], ignore_index=True)
    updated_df.to_csv(outname, sep=",", index=False)



def query(client, tweets, n, outname):
    """
    Query Claude API to classify a batch of tweets using iterative prompting.

    First prompts for step-by-step reasoning, then asks for structured yes/no/unsure answers.

    Args:
        client: Anthropic API client
        tweets: List of tweet texts to classify
        n: Number of repetitions for each tweet batch
        outname: Output CSV file path
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."
    prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"
    
    if os.path.isfile(outname):
        og_df = pd.read_csv(outname)
    else:
        og_df = pd.DataFrame({"tweet": [], "claude label": []})

    tweetlist = []
    labellist = []

    for t in tweets:
        prompt += "<tweet>%s</tweet>\n" % t

    # Repeat query n times to get multiple predictions per tweet
    for _ in range(n):
        retry = True
        n_retries = 0
        while retry:
            try:
                response = client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=1000,
                            system=context,
                            messages=[{"role": "user",
                                       "content": prompt}]
                                )
                message = response.content[0].text

                # Second step: get structured answer based on reasoning
                prompt2 = "Based on your reasoning above, answer the question in one word by saying \"yes\",\"no\", or \"unsure\" once for each tweet, where \"yes\" means that the tweet refers to opioids. Separate your answers by commas. Only give this in your response; do not add other content."
                time.sleep(0.25)
                response = client.messages.create(
                            model = "claude-sonnet-4-5",
                            max_tokens=24,
                            system=context,
                            messages=[{"role": "user",
                                       "content": prompt},
                                      {"role": "assistant",
                                       "content": message},
                                      {"role": "user",
                                       "content": prompt2}]
                                )
                message = response.content[0].text
            except anthropic.InternalServerError:
                if n_retries >= 10:
                    labels = ["APIError"] * len(tweets)
                    retry = False
                else:
                    time.sleep(1.5)
                    print("retrying...")
                    retry = True
                    n_retries += 1
            else:
                labels = message.split(",")
                retry = False
        if len(labels) != len(tweets):
            print("ERROR: mismatch in labels and tweets. aborting")
            print(len(tweets))
            print(message)
            return
        for i in range(len(tweets)):
            tweetlist.append(tweets[i])
            labellist.append(labels[i])
        time.sleep(0.25)
        
    df = pd.DataFrame({"tweet": tweetlist, "claude label": labellist})
    updated_df = pd.concat([og_df, df], ignore_index=True)
    updated_df.to_csv(outname, sep=",", index=False)


def main(args):
    """
    Main execution function for tweet classification with Claude.

    Reads tweets from input CSV and processes them in batches.
    Results are appended to output CSV.
    """
    load_dotenv()
    client = anthropic.Anthropic()

    # Read input CSV
    if args.header:
        df = pd.read_csv(args.incsv)
        all_tweets = df["tweet"]
    else:
        df = pd.read_csv(args.incsv, header=None)
        all_tweets = df[0]

    # Load previously processed tweets to avoid duplicates
    prev_tweets = []
    if os.path.isfile(args.outname):
        prev_tweets = pd.read_csv(args.outname)["tweet"].unique().tolist()

    tweets = []
    tweets_per_query = 3
    query_repetitions = 5
    for tweet in tqdm(all_tweets):
        if tweet in prev_tweets:
            continue
        tweets.append(tweet)
        if len(tweets) == tweets_per_query:
                query(client, tweets, query_repetitions, args.outname)
                tweets = []
    if len(tweets) > 0:
        query(client, tweets, query_repetitions, args.outname)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incsv", type=str, help="path to input csv")
    parser.add_argument("--outname", type=str, help="path to output file")
    parser.add_argument("--header", action="store_true", help="flag for if there is a header in the input csv")
    args = parser.parse_args()
    main(args)
