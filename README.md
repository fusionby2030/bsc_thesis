# B.Sc. Thesis Universität Leipzig 2021

Topic: Analysis of machine learning towards predicting plasma density using the JET pedestal database.

Advisor(s): Frank Cichos & Mathias Groth & Aaro Järvinen

### Outline of repository 

- main thesis document, `thesis.pdf`, and presentation showing main results `pres.pdf` are found in the home dir. 
- `src/` holds the code, where experiment scripts are found locally, and reused methods like dataprocessing are found in `src/codebase/`
- `/src/out/` holds the data used for plotting found in the thesis 
- `doc/` is for latex
- `etc/` is for brainstorming 

Experiments output either a `.pickle` or `.txt` file, but the specifics are found in each script. Most experiments use arguement parsing so you can throw everything in a bash script and send it to your favourite cluster. 
