from aligner import SentenceAligner

model = SentenceAligner()
result = model.get_word_aligns(["Sir Nils Olav III. was knighted by the norwegian king .",
                       "Nils Olav der Dritte wurde vom norwegischen König zum Ritter geschlagen ."])
print(result)
