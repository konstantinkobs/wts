import os
import json
import shutil
import urllib
import gzip

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DOWNLOAD_SEMANTIC_SCHOLAR = True


def download_semantic_scholar_dataset(download_path: str) -> None:
    sem_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2019-01-31/"
    print("\tDownloading semantic scholar first. ")
    os.mkdir(download_path)
    with urllib.request.urlopen(sem_url + "manifest.txt") as response, open(download_path + "manifest.txt", 'wb') as fh:
        shutil.copyfileobj(response, fh)
    with open(download_path + "/manifest.txt", "r") as f:
        for line in tqdm(f):
            line = line.strip()
            with urllib.request.urlopen(sem_url + line) as response, open(download_path + line, 'wb') as fh:
                shutil.copyfileobj(response, fh)
            if "s2-corpus-" in line:
                with gzip.open(download_path + line, 'rb') as f_in:
                    with open(download_path + line[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(download_path + line)


def stream_whole_computer_science_dataset(corpus_path: str, output_path: str, venue_path: str, verbose: int = 1)\
        -> None:
    """
    :param corpus_path:
    :param output_path:
    :param venue_path:
    :param verbose:
    :return:
    """
    if verbose:
        print("Loading computer science dataset ...")
    mismatch_counter, exact_match_counter, missing_info_counter, pub_counter, non_ai_counter = \
        0, 0, 0, 0, 0
    with open(venue_path, "r") as f:
        venues = f.readlines()
    venues = venues[0].strip().split(";")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for file in tqdm(os.listdir(corpus_path)):
            if "s2-corpus-" in file:
                with open(corpus_path + file, 'r', encoding='utf-8') as f:
                    for l in f:
                        ret_val = {}
                        tmp = json.loads(l)
                        for venue in venues:
                            matched = False
                            if venue in tmp["venue"] and not matched:
                                pub_counter += 1
                                matched = True
                                ret_val['venue'] = venue
                                ret_val['title'] = tmp['title']
                                ret_val['keywords'] = tmp['entities']
                                ret_val['abstract'] = tmp['paperAbstract']
                                if tmp['paperAbstract'] and tmp['entities']:
                                    json.dump(ret_val, out_f)
                                    out_f.write("\n")
                                    if venue is not tmp["venue"]:
                                        mismatch_counter += 1
                                    else:
                                        exact_match_counter += 1
                                else:
                                    missing_info_counter += 1
                            elif 'EC' in tmp["venue"] and non_ai_counter < 20000 and not matched:
                                matched = True
                                ret_val['venue'] = "non_ai"
                                ret_val['title'] = tmp['title']
                                ret_val['keywords'] = tmp['entities']
                                ret_val['abstract'] = tmp['paperAbstract']
                                if tmp['paperAbstract'] and tmp['entities']:
                                    json.dump(ret_val, out_f)
                                    out_f.write("\n")
                                    non_ai_counter += 1
    print("Finished. ")


def stream_medline_dataset(corpus_path: str, output_path: str, verbose: int = 1) -> None:
    """
    :param corpus_path:
    :param output_path:
    :param verbose:
    :return:
    """
    if verbose:
        print("Extracting all venues from medline. ")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for file in tqdm(os.listdir(corpus_path)):
            if "s2-corpus-" in file:
                with open(corpus_path + file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = json.loads(line)
                        if "Medline" in line['sources'] and 'venue' in line:
                            tmp = {"title": line["title"], "venue": line["venue"]}
                            if line['paperAbstract']:
                                tmp["abstract"] = line["paperAbstract"]
                            if line["entities"]:
                                tmp["keywords"] = line["entities"]
                            json.dump(tmp, out_f)
                            out_f.write("\n")
    print("Finished extracting Medline. ")


def stream_top_k_semantic_scholar(corpus_path: str, output_path: str, top_k: int = 100, verbose: int = 1):
    """
    :param corpus_path:
    :param output_path:
    :param top_k:
    :param verbose:
    :return:
    """
    venues = {}
    for file in tqdm(os.listdir(corpus_path)):
        if "s2-corpus-" in file:
            with open(corpus_path + file, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = json.loads(line)
                    if "title" in line and "entities" in line and "paperAbstract" in line:
                        venue = line['venue']
                        if "@" in venue:
                            venue = venue.split("@")[-1]
                        if "(" in venue:
                            venue = venue.split("(")[1][:-1]
                        if venue in venues:
                            venues[venue] += 1
                        else:
                            venues[venue] = 1
    venues = sorted([(venue, count) for venue, count in venues.items()], key=lambda x: x[1], reverse=True)[:top_k]
    venues = [x[0] for x in venues]
    if verbose:
        print("Extracted all venues. Saving to ", output_path, "venues.csv")
    with open(output_path + "venues.csv", "w", encoding="utf-8") as f:
        f.write(";".join(venues))
    with open(output_path + "semantic_scholar.json", "w", encoding="utf-8") as out_f:
        for file in tqdm(os.listdir(corpus_path)):
            if "s2-corpus-" in file:
                with open(corpus_path + file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        line = json.loads(line)
                        if "title" in line and "entities" in line and "paperAbstract" in line:
                            matched = False
                            tmp = {'title': line['title'], 'keywords': line['entities'], 'abstract': line['paperAbstract']}
                            for venue in venues:
                                if venue in line['venue'] and not matched:
                                    tmp['venue'] = venue
                                    matched = True
                                    json.dump(tmp, out_f)
                                    out_f.write("\n")
    print("Finished. ")


def remove_small_y_occurences(path_to_pubs, venues: list, percentage: int = None, top_k: int = None,
                              hard_cut_off: int = None) -> list:
    """
    :param path_to_pubs:
    :param venues:
    :param percentage:
    :param top_k:
    :param hard_cut_off:
    :return:
    """
    print("Loaded Venues. Now creating ranking. ")
    df = pd.DataFrame(data=venues, columns=["venue", "count"]).groupby("venue").agg("count").\
        sort_values(by=["count"], ascending=False)
    print("Ranking finished. ")
    print(df)
    if percentage:
        y_values = df.index.tolist()[:(100 - percentage)]
    elif top_k:
        y_values = df.index.tolist()[:top_k]
    else:
        count = len([x[0] for x in df.values.tolist() if x[0] >= hard_cut_off])
        y_values = df.index.tolist()[:count]
    pubs = []
    print("Extracting pubs")
    with open(path_to_pubs + ".json", 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            line = json.loads(line)
            if line['venue'] in y_values and "abstract" in line and "keywords" in line:
                pubs.append(line)
    return pubs


def create_stratified_train_test_split(path_to_pubs: str, output_addon: str = "", removal_method: str = "top_k"):
    """
    :param path_to_pubs:
    :param output_addon:
    :param removal_method: Either percentage, top_k or cutoff
    :return:
    """
    if path_to_pubs.endswith(".json"):
        path_to_pubs = path_to_pubs[:-5]
    venues = []
    print("Loading venues. ")
    with open(path_to_pubs + ".json", "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if "abstract" in line and "keywords" in line:
                venues.append((json.loads(line)['venue'], 1))
    if removal_method == "percentage":
        pubs = remove_small_y_occurences(path_to_pubs, venues, percentage=5)
    elif removal_method == "top_k":
        pubs = remove_small_y_occurences(path_to_pubs, venues, top_k=78)
    elif removal_method == "cutoff":
        pubs = remove_small_y_occurences(path_to_pubs, venues, hard_cut_off=20000)
    else:
        print("Removal Method unknown. ")
        exit(1)
    print("Created ", len(pubs), "pubs. Saving them. ")
    with open(path_to_pubs + "_reduced" + output_addon + ".json", "w", encoding="utf-8") as f:
        for line in pubs:
            json.dump(line, f)
            f.write("\n")
    y = [line['venue'] for line in pubs]
    x = pubs
    for line in x:
        del line['venue']
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    train = []
    for i, d in enumerate(x_train):
        d['venue'] = y_train[i]
        train.append(d)
    x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    test, val = [], []
    for i, d in enumerate(x_val):
        d['venue'] = y_val[i]
        val.append(d)
    for i, d in enumerate(x_test):
        d['venue'] = y_test[i]
        test.append(d)
    with open(path_to_pubs + "_train" + output_addon + ".json", "w", encoding="utf-8") as f:
        for line in train:
            json.dump(line, f)
            f.write("\n")
    with open(path_to_pubs + "_test" + output_addon + ".json", "w", encoding="utf-8") as f:
        for line in test:
            json.dump(line, f)
            f.write("\n")
    with open(path_to_pubs + "_val" + output_addon + ".json", "w", encoding="utf-8") as f:
        for line in val:
            json.dump(line, f)
            f.write("\n")


def load_stopwords(path: str = "../data/stopwords.txt") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines()]


def analyse_pubs(path_to_pubs: str, item_to_analyse: str = "venue"):
    """
    :param path_to_pubs:
    :param item_to_analyse: venue, title, abstract, keywords
    :return:
    """
    pubs = []
    with open(path_to_pubs, "r", encoding="utf-8") as f:
        for line in f:
            pubs.append(json.loads(line))
    if item_to_analyse is "venue":
        print("Analysing venues. ")
        data = [(x['venue'], 1) for x in pubs]
        return pd.DataFrame(data=data, columns=["data", "count"]).groupby("data").agg("sum"). \
            sort_values(by=['count'], ascending=False)
    elif item_to_analyse is "title":
        print("Analysing titles. ")
        avg = [len(pub['title'].strip().split(" ")) for pub in pubs]
        print("Average Length of titles: ", (sum(avg)/len(avg)))
        stopwords = load_stopwords()
        data = [[(word, 1) for word in pub['title'].strip().lower().split(" ") if word not in stopwords] for pub in pubs]
        return pd.DataFrame(data=[a for b in data for a in b], columns=["Word", "count"]).groupby("Word").agg("sum"). \
            sort_values(by=['count'], ascending=False)
    elif item_to_analyse is "abstract":
        print("Analysing abstract. ")
        avg = [len(pub['abstract'].strip().split(" ")) for pub in pubs if "abstract" in pub]
        print("Average Length of abstracts: ", (sum(avg)/len(avg)))
        stopwords = load_stopwords()
        data = [[(word, 1) for word in pub['abstract'].strip().lower().split(" ") if word not in stopwords]
                for pub in pubs if "abstract" in pub]
        return pd.DataFrame(data=[a for b in data for a in b], columns=["Word", "count"]).groupby("Word").agg("sum"). \
            sort_values(by=['count'], ascending=False)
    elif item_to_analyse is "keywords":
        print("Analysing keywords. ")
        avg = [len(pub['keywords']) for pub in pubs if "keywords" in pub]
        print("Average keywords: ", (sum(avg) / len(avg)))
        stopwords = load_stopwords()
        data = [[(word.lower(), 1) for word in pub['keywords'] if word.lower() not in stopwords]
                for pub in pubs if "keywords" in pub]
        return pd.DataFrame(data=[a for b in data for a in b], columns=["Word", "count"]).groupby("Word").agg("sum"). \
            sort_values(by=['count'], ascending=False)
    else:
        print("Item to analyse not implemented yet. ")
        return


def analyse_extracted_corpus(path_to_file: str, output_path: str) -> None:
    """
    :param path_to_file:
    :param output_path:
    :return:
    """
    with open(path_to_file, "r", encoding="utf-8") as in_f, open(output_path + ".json", "w", encoding="utf-8") as out_f:
        tmp = {"no_abstract": 0, "no_keywords": 0, "no_title": 0}
        for pub in tqdm(in_f):
            pub = json.loads(pub)
            if not pub['entities']:
                tmp["no_keywords"] = tmp["no_keywords"] + 1
            if not pub['paperAbstract']:
                tmp["no_abstract"] = tmp["no_abstract"] + 1
            if not pub['title']:
                tmp['no_title'] = tmp['no_title'] + 1
            if pub['venue'] not in tmp:
                tmp[pub['venue']] = 1
            else:
                tmp[pub['venue']] = tmp[pub['venue']] + 1
            out_dict = {"venue": pub['venue'], "abstract": pub['paperAbstract'].strip(), "title": pub['title'].strip(),
                        "keywords": pub['entities']}
            json.dump(out_dict, out_f)
            out_f.write("\n")
    print("Finished. ", tmp)
    with open(output_path + "_described.json", "w", encoding="utf-8") as f:
        json.dump(tmp, f)
        f.write("\n")


if __name__ == '__main__':
    CORPUS_PATH = "../data/data/"
    if DOWNLOAD_SEMANTIC_SCHOLAR:
        print("Downloading semantic scholar. ")
        download_semantic_scholar_dataset(download_path=CORPUS_PATH)
    print("Loading CS dataset. ")
    stream_whole_computer_science_dataset(corpus_path=CORPUS_PATH, venue_path=CORPUS_PATH + "venues.txt",
                                          output_path=CORPUS_PATH + "computer_science/computer_science.json")
    print("Split CS dataset. ")
    create_stratified_train_test_split(path_to_pubs=CORPUS_PATH + "computer_science/computer_science.json")
    print("Loading Medline dataset. ")
    stream_medline_dataset(corpus_path=CORPUS_PATH, output_path=CORPUS_PATH + "medline/medline.json")
    print("Split Medline dataset. ")
    create_stratified_train_test_split(path_to_pubs=CORPUS_PATH + "medline/medline.json")
    print("##### Computer Science #####")
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "computer_science/computer_science_reduced.json", item_to_analyse="venue"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "computer_science/computer_science_reduced.json", item_to_analyse="title"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "computer_science/computer_science_reduced.json", item_to_analyse="abstract"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "computer_science/computer_science_reduced.json", item_to_analyse="keywords"))
    print("##### Medline #####")
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "medline/medline_reduced.json", item_to_analyse="venue"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "medline/medline_reduced.json", item_to_analyse="title"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "medline/medline_reduced.json", item_to_analyse="keywords"))
    print(analyse_pubs(path_to_pubs=CORPUS_PATH + "medline/medline_reduced.json", item_to_analyse="abstract"))
    print("Loading Semantic scholar dataset. ")
    stream_top_k_semantic_scholar(corpus_path=CORPUS_PATH, output_path=CORPUS_PATH + "semantic_scholar/")
