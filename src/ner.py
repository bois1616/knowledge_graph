# -*- coding: UTF-8 -*-

# import
import pandas as pd
import numpy as np
from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import uuid
import json
import ollama.client as client

# globals

# functions


def row_to_named_entities(row):
    # print(row)
    ner_results = ner(row['text'])
    metadata = {'chunk_id': row['chunk_id']}
    entities = []
    for result in ner_results:
        entities = entities + [{'name': result['word'], 'entity': result['entity_group'], **metadata}]

    return entities


def dataframe_text_to_named_entities(dataframe) -> pd.DataFrame:
    # Takes a dataframe from the parsed data and returns dataframe with named entities.
    # The input dataframe must have a text and a chunk_id column.

    assert 'text' in dataframe.columns, "The dataframe must have a text column."
    assert 'chunk_id' in dataframe.columns, "The dataframe must have a chunk_id column."

    # Using swifter for parallelism
    # 1. Calculate named entities for each row of the dataframe.
    results: pd.DataFrame = dataframe.apply(func=row_to_named_entities, axis=1)

    ## Flatten the list of lists to one single list of entities.
    entities_list: list = np.concatenate(results).ravel().tolist()

    ## Remove all NaN entities
    entities_dataframe = pd.DataFrame(entities_list).replace(' ', np.nan)
    entities_dataframe = entities_dataframe.dropna(subset=['entity'])

    ## Count the number of occurances per chunk id
    entities_dataframe = (entities_dataframe.groupby(['name', 'entity', 'chunk_id']).
                          size().
                          reset_index(name='count')
                          )

    return entities_dataframe

def extract_concepts(prompt: str, model='mistral-openorca:latest'):
    SYS_PROMPT = (
        "Your task is to extract the key entities mentioned in the users input.\n"
        "Entities may include - event, concept, person, place, object, document, organisation, artifact, misc, etc.\n"
        "Format your output as a list of json with the following structure.\n"
        "[{\n"
        "   \"entity\": The Entity string\n"
        "   \"importance\": How important is the entity given the context on a scale of 1 to 5, 5 being the highest.\n"
        "   \"type\": Type of entity\n"
        "}, { }]"
    )
    response, context = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    return json.loads(response)


if __name__ == '__main__':
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
    )

    ## Roberta based NER
    # ner = pipeline("token-classification", model="2rtl3/mn-xlm-roberta-base-named-entity", aggregation_strategy="simple")
    ner = pipeline("token-classification",
                   model="dslim/bert-large-NER",
                   aggregation_strategy="simple",
                   )

    print("Number of parameters ->", ner.model.num_parameters() / 1_000_000, "Mn")

    loader = PyPDFLoader("./data/GlobalPublicHealth2022.pdf")
    # loader = PyPDFDirectoryLoader("./data/kesy1dd")

    pages = loader.load_and_split(text_splitter=splitter)
    print(f'{len(pages)=}')

    rows = []
    for page in pages:
        row = {'text': page.page_content, **page.metadata, 'chunk_id': uuid.uuid4().hex}
        rows += [row]

    df = pd.DataFrame(rows)

    dfne: pd.DataFrame = dataframe_text_to_named_entities(df)
    df_ne = (dfne.groupby(['name', 'entity']).
             agg({'count': 'sum', 'chunk_id': ','.join}).
             reset_index()
             )
    df_ne.sort_values(by='count', ascending=False).head(100).reset_index()

    print(f'{pages[12].page_content=}')

    res = extract_concepts(prompt=pages[22].page_content)

    print(f'{res= }')

