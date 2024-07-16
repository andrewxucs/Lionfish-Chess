<img width="293" length="800" alt="Lionfish graphic" src="https://github.com/user-attachments/assets/736a587d-8ec6-47f6-92c8-6b89bd5e72a6">


# Introduction
Lionfish is a simple chess engine programmed in python that is stronger than the vast majority of human players\
The central algorithm of Lionfish is MTD-bi search\
The name of this engine is inspired by Stockfish, the strongest open-source engine in the world.

# User Interface
The board is represented with 64 characters.\
A square with no pieces contains '.'\
P, R, N, B, Q, K represents the pawn, rook, knight, bishop, queen, and king respectively\
The user's pieces are upper case, while Lionfish's pieces are lower case\
For simplicity, moves are characterized by the starting and ending coordinates of the move. For instance, moving the king's pawn two squares up is e2e4

# Limitations
Lionfish is not strong enough to defeat professional chess players\
Users only have the option to play the white pieces\
The engine is only compatible with Python 3\
Lionfish does not support underpromotion (promotion of a pawn to any piece other than a queen)

# Credits
The piece-squared tables were compiled by Thomas Ahle using Stockfish\
The MTD-bi search algorithm was discovered by Aske Plaat at the University of Alberta\
The relative piece values were calculated by AlphaZero and published by Google Deepmind in a research paper
