import csv
from typing import Iterable, Any

import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm

from rootflow.datasets.base import StreamDataset, DataItem


class WikipediaRaw(StreamDataset):
    _WIKIPEDIA_ROOT = "https://en.wikipedia.org"
    _INDEX_PAGE = (
        "/w/index.php?title=Category:All_Wikipedia_articles_written_in_American_English"
    )
    _LINKS_CONTAINER_CLASS = "mw-category mw-category-columns"
    _NEXT_PAGE_CONTAINER_CLASS = "mw-pages"
    _LINKS_FILENAME = "links.csv"

    def prepare_data(self, directory: str) -> Iterable[Any]:
        csv_filepath = os.path.join(directory, self._LINKS_FILENAME)
        with open(csv_filepath, "r") as csv_file:
            items = list(csv.reader(csv_file))[1:]
        return [item[0] for item in items]

    def download(self, directory: str) -> None:
        previous_link = None
        link = self._WIKIPEDIA_ROOT + self._INDEX_PAGE
        article_links = []
        for _ in tqdm(range(240)):
            response = requests.get(link)
            soup = BeautifulSoup(response.text, features="html.parser")
            article_list = soup.find("div", {"class": self._LINKS_CONTAINER_CLASS})
            articles = article_list.find_all("li")
            for article in articles:
                article_links.append(article.find("a")["href"])
            next_page = soup.find(
                "div", {"id": self._NEXT_PAGE_CONTAINER_CLASS}
            ).find_all("a")[-1]["href"]
            next_link = self._WIKIPEDIA_ROOT + next_page
            if next_link == previous_link:
                break
            else:
                previous_link = link
                link = next_link

        csv_filepath = os.path.join(directory, self._LINKS_FILENAME)
        with open(csv_filepath, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["link"])
            for link in article_links:
                writer.writerow([link])

    def fetch_item(self, address: Any) -> DataItem:
        item_link = self._WIKIPEDIA_ROOT + address
        response = requests.get(item_link)
        return DataItem(data=response.text, id=address)


if __name__ == "__main__":
    dataset = WikipediaRaw("data")
    for item in tqdm(dataset):
        pass
