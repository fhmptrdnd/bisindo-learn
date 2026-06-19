import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print(f"Ditemukan {len(voices)} suara di komputermu:\n")

for i, voice in enumerate(voices):
    print(f"Suara {i+1}:")
    print(f" - Nama: {voice.name}")
    print(f" - ID: {voice.id}")
    print("-" * 30)