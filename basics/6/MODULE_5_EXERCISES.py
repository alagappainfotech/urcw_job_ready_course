"""
Module 5: OOP & Advanced Features - Exercises and Labs
"""

# HTML classes lab
class HTMLElement:
    tag = ""
    def __init__(self, children=None, text=""):
        self.children = children or []
        self.text = text
    def __str.me__(self):  # placeholder to check syntax later
        return ""
    def __str__(self):
        inner = self.text + "".join(str(c) for c in self.children)
        return f"<{self.tag}>" + inner + f"</{self.tag}>"

class Html(HTMLElement):
    tag = "html"

class Body(HTMLElement):
    tag = "body"

class P(HTMLElement):
    tag = "p"


def lab_html_document_demo():
    doc = Html([Body([P(text="Hello"), P(text="World")])])
    print(str(doc))


# TypedList exercise
class TypedList:
    def __init__(self, item_type, initial=None):
        self.item_type = item_type
        self._data = []
        for v in (initial or []):
            self.append(v)
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]
    def __delitem__(self, idx):
        del self._data[idx]
    def append(self, value):
        if not isinstance(value, self.item_type):
            raise TypeError("wrong item type")
        self._data.append(value)


# StringDict lab
class StringDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        temp = dict(*args, **kwargs)
        for k, v in temp.items():
            self[k] = v
    def __setitem__(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("StringDict only accepts string keys and values")
        return super().__setitem__(key, value)


# Regex phone normalizer
import re
def normalize_phone(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        country = "1"
        area, exch, last = digits[:3], digits[3:6], digits[6:]
    elif len(digits) == 11 and digits[0] == "1":
        country = digits[0]
        area, exch, last = digits[1:4], digits[4:7], digits[7:]
    else:
        raise ValueError("invalid phone")
    if area[0] in "01" or exch[0] in "01" or area[1] == "9":
        raise ValueError("invalid NANP area/exchange")
    return f"{country}-{area}-{exch}-{last}"


if __name__ == "__main__":
    lab_html_document_demo()
    tl = TypedList(int, [1, 2, 3])
    print("TypedList length:", len(tl))
    d = StringDict({"a": "1"})
    print("StringDict:", d)
    print(normalize_phone("(212) 555-1212"))


