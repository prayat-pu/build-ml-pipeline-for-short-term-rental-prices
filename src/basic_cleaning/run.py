#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def cleaning_for_price(df,min_price,max_price):
    # Drop outliers
    idx = df['price'].between(min_price,max_price)
    df = df[idx].copy()
    return df


def cleaning_for_last_review(df):
    # covert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    return df

def cleaning(df,args):
    df = cleaning_for_price(df,args.min_price,args.max_price)
    df = cleaning_for_last_review(df)
    return df


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    ######################
    # YOUR CODE HERE     #
    ######################
    # download raw data from the wandb
    local_path = wandb.use_artifact(args.input_artifact).file()

    df = pd.read_csv(local_path)
    cleaned_df = cleaning(df.copy(),args)
    cleaned_df.to_csv(args.output_artifact,index=False)


    # upload cleaned data into artifact in wabdb
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help='filename of the input artifact', ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help='filename of output artifact', ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help='type of output artifact', ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str, ## INSERT TYPE HERE: str, float or int,
        help='description of output', ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float, ## INSERT TYPE HERE: str, float or int,
        help='minimum price', ## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float, ## INSERT TYPE HERE: str, float or int,
        help='maximum price', ## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
