from src.disfluency_generation import LARD
from src.embeddings_generator import EmbeddingGenerator
from src.create_dataset import create_dataset

if __name__ == "__main__":
    # Initialize lard tool
    lard = LARD()
    eg = EmbeddingGenerator()

    # Generate repetitions
    # fluent_sentence = "Hello hello"
    # # This is a first-degree repetition
    # disfluency = lard.create_repetitions(fluent_sentence, 1, is_fluent=False, existing_annotations=['D', 'F'])
    # print(disfluency)
    # assert False
    #
    # # This is a second-degree repetition
    # disfluency = lard.create_repetitions(fluent_sentence, 2)
    # print(disfluency)
    #
    # #  Generate replacements
    # fluent_sentence = "Yes, I am going to visit my family for a week"
    # disfluency = lard.create_replacements(eg, fluent_sentence)
    # print(disfluency)
    #
    fluent_sentence = "I would like to eat pancakes for breakfast."
    disfluency = lard.create_replacements(eg, fluent_sentence)
    print(disfluency)

    # # # Generate restarts
    # fluent_sentence_1 = "where can i find a pharmacy near me ?"
    # fluent_sentence_2 = "what time do you close ?"
    # disfluency = lard.create_restarts(fluent_sentence_1, fluent_sentence_2)
    # print(disfluency)

    create_dataset('/home/tatiana/PycharmProjects/disfl_qa/train.csv', output_dir='/home/tatiana/PycharmProjects/artificial-disfluency-generation/data/output_data/difsl-qa', column_text='original',
                   percentages=[0, 0, 100])
