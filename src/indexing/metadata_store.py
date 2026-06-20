class MetadataStore:
    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def get(self, item_id: str) -> dict:
        return self._items.get(item_id, {})

    def upsert(self, item_id: str, metadata: dict) -> None:
        self._items[item_id] = {**self._items.get(item_id, {}), **metadata}
