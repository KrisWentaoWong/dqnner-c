## Document-level NER with Reinforcement Learning

This code project is implemented based on Karthik Narasimhan's work (https://github.com/karthikncode/DeepRL-InformationExtraction).

### Installation
You will need to install [Torch](http://torch.ch/docs/getting-started.html) and the  python packages in `requirements.txt`.  

You will also need to install the Lua dev library `liblua` (`sudo apt-get install liblua5.2`) and the [signal](https://github.com/LuaDist/lua-signal) package for Torch to deal with SIGPIPE issues in Linux.
(You may need to uninstall the [signal-fft](https://github.com/soumith/torch-signal) package or rename it to avoid conflicts.)

### Running the code
  * First copy the codes to '~/dqnner-c-lrs/', and run the server, for example:
    python ~/dqnner-c-lrs/Code/code/server.py

  * In a separate terminal/tab, then run the agent:
    cd ~/dqnner-c-lrs/Code/code/dqn
    ./run_cpu.sh CoNLLTest_RR_F4A SVM_LibSVM_RBF 50000 /mnt/hgfs/VMWareShare/20200428-Run3-lrs1[2.5e-05]/CoNLLTest_RR_F4A/SVM_LibSVM_RBF/HC=0.08/ 429 111 112 112 NER_C 20 20 1.0 5  0.08 0 0 None 55 2.5e-05 2.5e-05 1000000

  * NOTICE:
    Make sure that the following files are in the folder '/mnt/hgfs/VMWareShare/20190917-IDNER+ASK-sort-T7-JSON/CoNLLTest_RR_F4A/SVM_LibSVM_RBF/'.
    - train-q.docs.json
    - test.docs.json
    - test.goldPLOKeys.json
    - dev.docs.json
    - dev.goldPLOKeys.json
    These file can be found in our project: dqnner-c/Data.

  * If you have any question, don't hesitate to contact lutingming@163.com.

### Acknowledgements
  * [Karthik Narasimhan's DQN4IE codebase](https://github.com/karthikncode/DeepRL-InformationExtraction)
  * [Deepmind's DQN codebase](https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner)

