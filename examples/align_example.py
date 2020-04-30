import simalign

source_sentence = "Sir Nils Olav III. was knighted by the norwegian king ."
target_sentence = "Nils Olav der Dritte wurde vom norwegischen König zum Ritter geschlagen ."

model = simalign.SentenceAligner()
result = model.get_word_aligns(source_sentence.split(), target_sentence.split())
print(result)
