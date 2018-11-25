






class SymbolsManager():
    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            self.add_symbol('<S>')
            self.add_symbol('<E>')
            self.add_symbol('<U>')
