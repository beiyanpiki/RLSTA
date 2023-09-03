text = """
//model
mdp

module node
   state : [1..36] init 1;
   action : [1..70] init 1;

"""

import numpy as np

trans = np.load("transmatrix_final.npy")

for state in range(trans.shape[0]):
    line = f"[p{state+1}] state = {state+1} -> "
    sum = 0
    for next_state in range(trans.shape[2]):
        xx = int(trans[state][0][next_state] * 100) * 1.0 / 100
        line += f"   {xx if next_state != trans.shape[2]-1 else max(0,(1-sum)):.2f}: (action' = {next_state+1}) +"
        sum += xx
    if line[-1:] == "+":
        line = line[:-1]
    line += ";\n"

    text += line

print(text)

for state in range(trans.shape[0]):
    for action in range(trans.shape[1]):
        nxt = np.argmax(trans[state][action])
        line = f"   [p{state+1}] state = {state+1} & action = {action+1} -> (state' = {nxt+1});\n"
        text += line

text += 'endmodule\n\n\nrewards "safe"\n'

reward = np.load("rewardmatrix.npy")
for state in range(reward.shape[0]):
    for action in range(reward.shape[1]):
        if reward[state][action] == 1:
            continue
        line = f"   [p{state+1}] state = {state+1} & action = {action+1} : {reward[state][action]:3f};\n"
        text += line
text += "endrewards"


with open("output.pm", "w") as file:
    # Write data to the file
    file.write(text)
