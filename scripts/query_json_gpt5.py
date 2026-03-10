"""
Query tweets using OpenAI GPT-5 API to classify opioid-related content.

This script supports both batch and individual query modes. Batch mode generates
JSONL files for batch API processing. Individual mode processes tweets in real-time
using conversation context.
"""
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import pandas as pd
import argparse
import time
from tqdm import tqdm
import json


def batch_query(model, tweets, n, outputf):
    """
    Generate batch query file for OpenAI Batch API.

    Args:
        model: Model name to use
        tweets: List of tweet batches to process
        n: Number of repetitions for each tweet batch
        outputf: Output JSONL file path for batch requests
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."

    f = open(outputf, "w")
    for query_idx, query_tweets in enumerate(tweets):
        prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"

        for t in query_tweets:
            prompt += "<tweet>%s</tweet>\n" % t
        for repeat in range(n):
            r = {"custom_id": "tweet%d_%d" % (query_idx, repeat),
                     "method": "POST",
                     "url": "/v1/chat/completions",
                     "body": {"model": model,
                              "messages": [
                                  {"role": "system",
                                   "content": context},
                                  {"role": "user",
                                   "content": prompt}
                              ],
                              "max_tokens": 1000}
                    }
            f.write("%s\n" % json.dumps(r))
    f.close()


def batch_query_followup(model, tweets, responses, outputf):
    """
    Generate followup batch queries to extract structured answers from reasoning.

    Args:
        model: Model name to use
        tweets: List of tweet batches
        responses: List of JSON responses from initial batch query
        outputf: Output JSONL file path for followup batch requests
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."
    prompt2 = "Based on your reasoning above, answer the question in one word by saying \"yes\",\"no\", or \"unsure\" once for each tweet, where \"yes\" means that the tweet refers to opioids. Separate your answers by commas. Only give this in your response; do not add other content."

    # Parse responses and organize by query index
    response_dict = {}
    for r in responses:
        j = json.loads(r)
        query_id = j["custom_id"]
        query_idx = int(query_id.split("_")[0][5:])
        if not query_idx in response_dict.keys():
            response_dict[query_idx] = []
        response_dict[query_idx].append(j["response"]["body"]["choices"][0]["message"]["content"])

    f = open(outputf, "w")
    for query_idx, query_tweets in enumerate(tweets):
        prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"

        for t in query_tweets:
            prompt += "<tweet>%s</tweet>\n" % t

        for repeat, r in enumerate(response_dict[query_idx]):
            req = {"custom_id": "tweet%d_%d" % (query_idx, repeat),
                   "method": "POST",
                   "url": "/v1/chat/completions",
                   "body": {"model": model,
                            "messages": [
                                {"role": "system",
                                 "content": context},
                                {"role": "user",
                                 "content": prompt},
                                {"role": "assistant",
                                 "content": r},
                                {"role": "user",
                                 "content": prompt2}
                            ],
                            "max_tokens": 24}
                  }
            f.write("%s\n" % json.dumps(req))
    f.close()

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
        og_df = pd.DataFrame({"tweet": [], "gpt4 label": []})
    tweetlist = []
    labellist = []
    for response in followup_responses:
        j = json.loads(response)
        query_id = j["custom_id"]
        query_idx = int(query_id.split("_")[0][5:])
        query_tweets = tweets[query_idx]
        labels = j["response"]["body"]["choices"][0]["message"]["content"]
        assert len(query_tweets) == len(labels.split(","))
        for i in range(len(query_tweets)):
            tweetlist.append(query_tweets[i])
            labellist.append(labels.split(",")[i].strip())
    df = pd.DataFrame({"tweet": tweetlist, "gpt4 label": labellist})
    updated_df = pd.concat([og_df, df], ignore_index=True)
    updated_df.to_csv(outname, sep=",", index=False)
    

def query(client, tweets, n, outname):
    """
    Query GPT-5 API to classify a batch of tweets using conversational context.

    Uses GPT-5's conversation API to maintain context across prompts for iterative reasoning.

    Args:
        client: OpenAI API client
        tweets: List of tweet texts to classify
        n: Number of repetitions for each tweet batch
        outname: Output CSV file path
    """
    context = "You are an AI assistant that helps people find information. You are particularly hip with online slang and know everything about how people talk on social media platforms like Facebook, Twitter, Reddit, and TikTok."
    prompt = "I am going to give you a series of tweets, delimited with the xml tags <tweet></tweet>. For each tweet, I want you to tell me if the tweet is directly referring to opioid use. Reason through your answers step-by-step.\n\n"
    
    if os.path.isfile(outname):
        og_df = pd.read_csv(outname)
    else:
        og_df = pd.DataFrame({"tweet": [], "gpt5 label": []})

    tweetlist = []
    labellist = []

    for t in tweets:
        prompt += "<tweet>%s</tweet>\n" % t

    # Repeat query n times to get multiple predictions per tweet
    for _ in range(n):
        retry = True
        n_retries = 0

        # Create conversation with system context
        conversation = client.conversations.create(items=[{"type": "message",
                                                           "role": "system",
                                                           "content": context}])
        while retry:
            try:
                response = client.responses.create(
                            conversation = conversation.id,
                            model = "gpt-5-mini",
                            input = prompt
                                )
                message = response.output[1].content[0].text

                # Add reasoning to conversation history
                _ = client.conversations.items.create(conversation.id,
                                                      items=[{"type": "message",
                                                              "role": "user",
                                                              "content": prompt},
                                                             {"type": "message",
                                                              "role": "assistant",
                                                              "content": message}])

                # Second step: get structured answer based on reasoning
                prompt2 = "Based on your reasoning above, answer the question in one word by saying \"yes\",\"no\", or \"unsure\" once for each tweet, where \"yes\" means that the tweet refers to opioids. Separate your answers by commas. Only give this in your response; do not add other content."
                print(message)
                time.sleep(0.25)
                response = client.responses.create(
                            conversation = conversation.id,
                            model = "gpt-5-mini",
                            input = prompt2
                                )
                message = response.output[1].content[0].text
            except openai.PermissionDeniedError:
                labels = ["ContentRestrictionError"] * len(tweets)
                retry = False
            except (openai.RateLimitError, KeyError, openai.APIConnectionError, openai.APIError, IndexError):
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
        _ = client.conversations.delete(conversation.id)
        time.sleep(0.25)
        
    df = pd.DataFrame({"tweet": tweetlist, "gpt5 label": labellist})
    updated_df = pd.concat([og_df, df], ignore_index=True)
    updated_df.to_csv(outname, sep=",", index=False)


def main(args):
    """
    Main execution function for tweet classification with GPT-5.

    Reads tweets from input CSV and processes them in batches.
    Results are appended to output CSV.
    """
    load_dotenv()
    client = OpenAI(api_key=os.getenv("GPT5_API_KEY"))

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
