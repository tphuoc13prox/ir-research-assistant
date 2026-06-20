class TextCleaner:
    def clean(self, text: str) -> str:
        return " ".join(text.split())
