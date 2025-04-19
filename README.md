# AI Course Assignment

This repository contains solutions to Assignment 03 for the AI-2002 Artificial Intelligence course at the National University of Computer and Emerging Sciences, FAST School of Computing, Fall 2025. The assignment comprises three problems focused on graph algorithms, subgraph detection, and constraint satisfaction problems, implemented in Python.

# Project Overview

The assignment includes three problems, each addressing a unique challenge:

1. **Course Crisis (Swapping):**
A system to manage student course section swap requests using graph-based algorithms (Greedy, DFS, BFS). It prioritizes older students (by batch: 21, 22, 23, 24) and outputs a tabular swap schedule in Excel, sorted by roll number.

2. **Graph Classification:**
A program to detect and classify subgraphs (cliques, chains, stars, cycles) in an undirected graph, using DFS, BFS, and custom heuristics to prioritize larger or critical subgraphs.

3. **Ultimate Tic-Tac-Toe:** 
A Constraint Satisfaction Problem (CSP) solver for Ultimate Tic-Tac-Toe, implementing Backtracking Search with Forward Checking and Arc Consistency, featuring a visual GUI and performance analysis.

Each problem is implemented in a separate Python script, with associated datasets, outputs, and documentation stored in dedicated directories. The code adheres to the assignment’s guidelines, avoiding built-in libraries except where explicitly allowed (e.g., pandas for Excel output, networkx for graph operations, tkinter for GUI).

**Directory Details**

course_crisis/: Contains the swapping problem solution, a preprocessing script to sort requests by batch, input dataset, sorted dataset, and Excel output.

graph_classification/: Includes the subgraph detection solution, input graph data, and a summary output.

ultimate_tictactoe/: Holds the Tic-Tac-Toe CSP solver, optional game configurations, GUI screenshots/video, and performance analysis.



Acknowledgments

Developed for AI-2002 Assignment 03, Fall 2025, FAST School of Computing.
References for Ultimate Tic-Tac-Toe: bejofo.com/ttt, michaelxing.com.


Created by Ahmad (Roll Number: 22i-1288)
