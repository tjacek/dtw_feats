import actions.io

read_actions=actions.io.ActionReader()
print(type(read_actions))
actions=read_actions('seqs')
print(type(actions[0]))