import difflib


class TextErrorsParser:
    def __init__(self, transcript) -> None:
        self.transcript = transcript
        
    def _get_too_long_breaks(self, words: list) -> list[float]:  # TODO: Tests
        """Calculate longest breaks because they pollute some of our pace measurements"""
        breaks = []
        last_end = None

        for word in words:
            if not last_end:
                last_end = word.end
                continue

            break_len = word.start - last_end
            if break_len > 2:
                breaks.append(break_len)
            
            last_end = word.end

        return breaks

    @staticmethod
    def _count_syllables(word) -> int:
        # Polish vowels
        vowels = "aeiouyąęó"
        word = word.lower()
        syllable_count = sum(1 for char in word if char in vowels)
        return syllable_count

    def calculate_speech_pace(self) -> float:  # TODO: Tests
        """Calculate speech pace excluding long breaks that pollute the output."""
        TOO_FAST_SPEAKING = 160  # Word per minute

        words = self.transcript.words
        word_count = len(words)
        first_word_start = words[0].start
        last_word_end = words[-1].end

        breaks = self._get_too_long_breaks(words)
        # Subtract all to long breaks and substitute them with 0.5 second breaks
        speech_len = last_word_end - first_word_start - sum(breaks) + len(breaks) * 0.4
        
        words_per_minute = 60 * word_count / speech_len
        print(f"{words_per_minute = }")
        return words_per_minute
    
    def detect_repetitions(self) -> list[str]:  # TODO: Tests
        """Calculate similarity of all words with each other"""
        said_words = []
        repetitions = []

        string_words = [w.word for w in self.transcript.words]
        for word in string_words:
            for said_word in said_words:
                similarity = difflib.SequenceMatcher(None, word.lower(), said_word.lower()).ratio()

                # Similarity works well but most of the time, words have to start with same letter and not be connectors like `i` or `na` 
                if similarity > 0.7 and word[0].lower() == said_word[0].lower() and len(word) > 2:
                    repetitions.extend([word, said_word])
            
            said_words.append(word)

        print(f"{repetitions = }")
        return repetitions
    
    def count_numbers_per_sentence(self):
        """We count word as number if any character of it is a digit"""
        TOO_MANY_NUMBER_PER_SENTENCE_OVER = 3

        sentence_ends = [i for i, w in enumerate(self.transcript.text.split(" ")) if "." in w]
        contains_number = lambda word: any(letter.isdigit() for letter in word)

        number_count = 0
        sentences_num_count = []
        for i, word in enumerate(self.transcript.words):
            if contains_number(word.word):
                number_count += 1
            
            if i in sentence_ends:
                sentences_num_count.append(number_count)
                number_count = 0
        
        print(f"{sentences_num_count = }")
        return sentences_num_count

    def calculate_fog_index(self) -> float:
        word_count = len(self.transcript.words)
        sentence_count = len([s for s in self.transcript.text.split(".") if s])
        long_words_count = len([w for w in self.transcript.words if self._count_syllables(w.word) > 3])

        # FOG index equation fro wikipedia
        return 0.4 * ((word_count / sentence_count) + 100 * (long_words_count / word_count))
