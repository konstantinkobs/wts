import re
import json
import math
import torch.utils.data


class IterableConferenceDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, max_length_title, max_length_abstract, max_length_keywords, embedding_vectors,
                 output_vectors):
        super(IterableConferenceDataset).__init__()
        self.path = path
        self.f = None
        self.max_length_title = max_length_title
        self.max_length_abstract = max_length_abstract
        self.max_length_keywords = max_length_keywords
        self.embedding_vectors = embedding_vectors
        self.output_vectors = output_vectors

    def __iter__(self):
        if self.f:
            self.f.close()
        self.f = open(self.path)
        return self

    def __next__(self):
        line = self.f.readline()
        if line:
            j = json.loads(line)
            tmp_list = []
            for entity in [["title", self.max_length_title], ["abstract", self.max_length_abstract],
                           ["keywords", self.max_length_keywords]]:
                if entity[0] == "keywords" and isinstance(j[entity[0]], list):
                    tmp = re.split('[^\w-]+', " ".join(j["keywords"]).lower())
                else:
                    tmp = re.split('[^\w-]+', j[entity[0]].lower())
                tmp_indices = []
                for word in tmp:
                    try:
                        index = self.embedding_vectors[word]
                        tmp_indices.append(index)
                    except:
                        tmp_indices.append(0)
                tmp_indices = tmp_indices[0:min(len(tmp_indices), entity[1])] + \
                              ([len(self.embedding_vectors)] * max(entity[1] - len(tmp_indices), 0))
                tmp_list.append(tmp_indices)
            if "venue" in j:
                return torch.LongTensor(tmp_list[0]), torch.LongTensor(tmp_list[1]), \
                      torch.LongTensor(tmp_list[2]), self.output_vectors[j["venue"].lower()]
            else:
                return torch.LongTensor(tmp_list[0]), torch.LongTensor(tmp_list[1]), \
                       torch.LongTensor(tmp_list[2])
        else:
            raise StopIteration

    def get_amount_output_classes(self) -> int:
        return len(self.output_vectors)

    # Maybe we will need this to enable multiple worker
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_end = dataset.end
        per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        dataset.start = overall_start + worker_id * per_worker
        dataset.end = min(dataset.start + per_worker, overall_end)
