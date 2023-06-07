import sys

def extract(str,tag):
    sub1 = "<" + tag + ">"
    sub2 = "</"+ tag + ">"
    
    idx1 = str.index(sub1)
    idx2 = str.index(sub2)

    output = str[idx1 + len(sub1) + 1: idx2]
    return output

def main(argv):
    file1 = open('train.txt', 'r')
    file2 = open('train_annot.txt', 'r')
    file3 = open('examples.txt', 'a')
    lines1 = file1.readlines()[int(argv[0])-1].strip()
    lines2 = file2.readlines()[int(argv[1])-1].strip()

    pref = extract(lines1, "input")
    print(pref)
    pref_list = pref.split(" ")[0:6]

    lines1 = extract(lines1, "dialogue")
    lines2 = extract(lines2, "dialogue")

    # newline = ""
    newline = "This is an negotiation conversation. There are {} books, {} hats, and {} balls. People should negotiate to reach agreement on item distribution. ".format(pref_list[0],pref_list[2],pref_list[4])
    newline = newline + "Fill the rest of the conversation by following the similar format as provided. Conversation: "

    for i in range(0, len(lines1.split("<eos>"))):
        each1 =  lines1.split("<eos>")[i]
        if "<selection>" in each1:
            break
        else:
            each2 =  lines2.split("<eos>")[i]
            each2 = each2.split(": ")[1]
            # newline = newline + " utterance: " + each1 + " annotation: " + each2
            newline = newline + "utterance: " + each1 + "<eos> " + "annotation: " + each2 + "<eos> "
    file3.write(newline+"\n")

if __name__ == "__main__":
    main(sys.argv[1:])