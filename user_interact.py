# Python program for user to interact with the models through the command-line interface

from PyInquirer import prompt
from inference.lstm_encoder_decoder_inference import generate_title as generate_title_model_1
from inference.bi_lstm_encoder_lstm_decoder_inference import generate_title as generate_title_model_2
from inference.glove_bi_lstm_encoder_lstm_decoder_inference import generate_title as generate_title_model_3
from inference.reverse_input_bi_lstm_encoder_lstm_decoder_inference import generate_title as generate_title_model_4

def interact(): 
    while True: 
        print("Hi, I am here to come up with a title for your essay? ")

        # user_input = input("Hi, I am here to come up with a title for your essay? Type 'exit' to quit\n")
        # user_input = user_input.lower().strip() 
        # print("Which model do you want to try? Select from the below options")

        questions_0 = [
            {
                'type': 'list',
                'name': 'Source',
                'message': "Would you like to try a sample essay or provide your own?",
                "choices": ["I will provide a text file",
                     "I want to use the sample essays"
                ]
            }
        ]
        answers_0 = prompt(questions_0)['Source']

        if answers_0 == "I will provide a text file":
            questions_1 = [
                {
                    'type': 'input',
                    'name': 'TextFile',
                    'message': "Please type in the relative path to the .txt file containing your essay, type 'exit' to quit",
                }
            ]
            answers_1 = prompt(questions_1)['TextFile']
            if answers_1 == 'exit':
                break

            try: 
                ret = generate_title_from_sample(answers_1)
                if ret == '1':
                    break
            except FileNotFoundError:
                print("Wrong file or file path, please try again")
        else:
            questions_2 = [
                {
                    'type': 'list',
                    'name': 'Sample',
                    'message': "Choose your sample essay",
                    "choices": [
                        "Sample essay 1: This Project Proposal",
                        "Sample essay 2: Short essay on the movie 'Rear Window'",
                        "Sample essay 3: Short English essay on Beatlemania",
                        "Sample essay 4: An excerpt from an article in the Miscellany News",
                        "Exit"
                    ]
                }
            ]
            answers_2 = prompt(questions_2)['Sample']

            if answers_2 == "Sample essay 1: This Project Proposal":
                generate_title_from_sample("sample_essays/sample_1.txt")
            elif answers_2 == "Sample essay 2: Short essay on the movie 'Rear Window'":
                generate_title_from_sample("sample_essays/sample_2.txt")
            elif answers_2 == "Sample essay 3: Short essay English essay on Beatlemania":
                generate_title_from_sample("sample_essays/sample_3.txt")
            elif answers_2 == "Sample essay 4: An excerpt from an article in the Miscellany News":
                generate_title_from_sample("sample_essays/sample_4.txt")
            else: 
                break

    print("")
    print("Goodbye! I'm still learning to comprehend text better!")
    
        

def generate_title_from_sample(path_to_txt_file):
    questions = [
        {
            'type': 'list',
            'name': 'Model',
            'message': "Which model do you want to try? Select from the below options",
            "choices": ["LSTM encoder-decoder using one-hot encoding with Bahdanau attention",
                    "Bi-LSTM encoder - LSTM decoder one-hot encoding with Bahdanau attention",
                    "Bi-LSTM encoder - LSTM decoder GloVe embeddings with Bahdanau attention",
                    "Reversed Input Bi-LSTM encoder - LSTM decoder one-hot encoding with Bahdanau attention",
                    "Back"
            ]
        }
    ]
    answers = prompt(questions)['Model']
    
    if answers == 'LSTM encoder-decoder using one-hot encoding with Bahdanau attention':
        print(generate_title_model_1("{}".format(path_to_txt_file)))
    elif answers == 'Bi-LSTM encoder - LSTM decoder one-hot encoding with Bahdanau attention':
        print(generate_title_model_2("{}".format(path_to_txt_file)))
    elif answers == "Bi-LSTM encoder - LSTM decoder GloVe embeddings with Bahdanau attention":
        print(generate_title_model_3("{}".format(path_to_txt_file)))
    elif answers == "Reversed Input Bi-LSTM encoder - LSTM decoder one-hot encoding with Bahdanau attention":
        print(generate_title_model_4("{}".format(path_to_txt_file)))
    else: 
        return

if __name__ == '__main__':
    interact()