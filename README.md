# Quantum-Natural-Language-Processing-with-lambeq---Quantinuum
Womanium Quantum Hackathon 2022
Team name: Maria
Team members (1) : Maria Gragera Garces

##Repository organization
This repository is divided into various folder that contain the submissions for every task in the challenge.
The final submission, aka task 6, associates sentences with either food or coding, and distinguishes between positive or neutral sentiment.
The overall system is quite simplistic and relies on a solid pre-processing step, that categorizes the sentences association to those two elements through a state vector.
If the sentence is associate with food, and has positive language, the resulting state will be [0,1].
If the sentence is associate with food, and has neutral language, the resulting state will be [0,0].
If the sentence is associate with code, and has positive language, the resulting state will be [1,1].
If the sentence is associate with code, and has neutral language, the resulting state will be [1,0].
