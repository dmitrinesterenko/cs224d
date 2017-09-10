from rnn import Config, RNN_Model
import tree
from tree import Tree, print_root
import pdb

def make_trees(sentences):
    """Given a sentence it creates a tree to pass to the model predictor"""
    trees = []
    for s in sentences:
        tree = Tree(s)
        trees.append(tree)
        print_root(tree.root)

    return trees

def play(sentences):
    """Generate sentiment for a sentence given a pre-trained netweork"""
    config = Config()
    config.weights_path = "./weights/adam"
    config.train = False
    model = RNN_Model(config)
    my_data = make_trees(sentences)
    predictions, _ = model.predict(my_data, model.weights_path())
    print(predictions)

if __name__=="__main__":
    sentence1 = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 bad) (2 film))) (3 (2 with) (4 (3 (3 awful) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
    sentence2 = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
    sentence3 = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 terrible) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
    sentence4 = "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 good) (2 film))) (3 (2 with) (4 (3 (3 awful) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
    play([sentence1, sentence2, sentence3, sentence4])


