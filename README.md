Code and output of autogrid experiments for the tecnical review: <<todo_link_arxiv_url>>

# File structure: 
the first level of folders represent each performed experiment experiment:

* **PPO extended experiment**: PPO agent trained with box and multiple action dicrete action spaces with reward IncreasingFlatReward on the Grid2Op environment l2rpn\_icaps\_2021\_large 
* **Agent and reward comparison Experiment**: Multiple agents from stable baselines 3 trained with diferent rewards() and action spaces (box (B) ,multiple action discrete (D) and single action discrete (S).

Each experiment folder contains the following file structure: 
* files: **\*.py** the Autogrid configuration files to execute the experiments.
* file: **Result extractor.py** python script to parse the output and generate plots and tables into the plots folder.
* file: **split_env.py** python script to split the environment into training and eval.
* folder: **plots** contain al the general plots and tables aswell as scenario specific plots.
* folders: **agents\_\<reward name\>** : contains the evaluation output of the agents trained with that reward
  * folders: **\<agent name\>\_<action space\>** the output of that agent using that specific action space trained with the reward specified on the parent folder.
    * file: the overal **performance.txt**
    * folders: specific performance on each evaluation scenario with the **Scenario\_<scenario name\>** folder name.

# Understanding the numbers:

**Score**: The evaluation score is the output of L2RPNSandBoxScore class which computes "grid operation cost": the sum of costs due to: loss, redispatch, storage and curtailment, each computed separately and then added together. Lower score mean lower operation cost which is better. 

**Timesteps**: reflects the time the grid stays operational (not game-over), each timestep represents 5 minutes of real-time in the simulation.
	
**Score\/Timesteps**:Obtained those two metrics we will compute the score over the timestep, reflecting the operation cost per 5 minutes of operation, so we can see if the agents that manage to stay "alive" longer incur in bigger operational costs.


 # Experiment results:

This are the plots and table from each expriment **plots** folder.

 ## PPO extended experiment
 
 | \{\}       | Timesteps | Score     | Score/Timesteps |
|------------|-----------|-----------|-----------------|
| PPO\_B   | 611\.56   | 17133\.91 | 28\.02          |
| PPO\_D   | 43\.99    | 1439\.36  | 32\.72          |
| Do nothing | 529\.85   | 14556\.00 | 27\.47          |

## Agent and reward comparison Experiment

### Do nothing agents
| \{\}             | Timestep | Score     | Score/timestep |
|------------------|----------|-----------|----------------|
| Simple Reconnect | 576\.19  | 16103\.91 | 27\.94         |
| Do Nothing       | 666\.73  | 18149\.54 | 27\.22         |

### Timesteps
| \{\}      | Bridge  | Overflow | Combined | Distance | Economic | Episode  | Redisp   | average |
|-----------|---------|----------|----------|----------|----------|----------|----------|---------|
| A2C\_B  | 94\.52  | 62\.38   | 8\.24    | 101\.22  | 126\.23  | 41\.00   | 263\.02  | 99\.52  |
| A2C\_D  | 894\.62 | 344\.43  | 876\.59  | 1111\.86 | 1255\.45 | 97\.93   | 957\.88  | 791\.25 |
| A2C\_S  | 225\.84 | 1002\.83 | 1064\.94 | 1064\.02 | 1158\.37 | 1215\.94 | 1014\.57 | 963\.79 |
| DDPG\_B | 131\.47 | 29\.57   | 59\.49   | 36\.89   | 154\.25  | 14\.12   | 64\.83   | 70\.09  |
| DQN\_D  | 302\.05 | 317\.15  | 245\.32  | 557\.87  | 434\.61  | 205\.59  | 210\.92  | 324\.79 |
| DQN\_S  | 307\.64 | 457\.35  | 387\.56  | 72\.36   | 278\.09  | 386\.82  | 308\.71  | 314\.08 |
| PPO\_B  | 17\.39  | 356\.62  | 2\.08    | 214\.98  | 311\.28  | 83\.44   | 77\.53   | 151\.90 |
| PPO\_D  | 100\.78 | 372\.71  | 413\.26  | 392\.24  | 329\.86  | 1136\.86 | 506\.67  | 464\.63 |
| PPO\_S  | 144\.31 | 284\.13  | 24\.91   | 270\.83  | 333\.62  | 141\.81  | 761\.79  | 280\.20 |
| Average   | 246\.51 | 358\.57  | 342\.49  | 424\.70  | 486\.86  | 369\.28  | 462\.88  |         |

![Timesteps](Agent%20and%20reward%20comparison%20Experiment/plots/Mean_timesteps.png?raw=true "Timesteps")

### Score
| \{\}      | Bridge     | Overflow   | Combined   | Distance   | Economic   | Episode    | Redisp     | Average    |
|-----------|------------|------------|------------|------------|------------|------------|------------|------------|
| A2C\_B  | 96887\.72  | 89497\.55  | 6060\.01   | 77400\.84  | 113410\.58 | 38019\.32  | 266808\.06 | 98297\.73  |
| A2C\_D  | 264591\.68 | 277667\.99 | 78676\.79  | 32849\.48  | 31593\.36  | 100109\.52 | 37517\.70  | 117572\.36 |
| A2C\_S  | 237641\.96 | 38712\.79  | 26423\.18  | 40943\.80  | 32757\.87  | 46531\.48  | 34588\.34  | 65371\.35  |
| DDPG\_B | 159700\.77 | 19125\.33  | 78250\.08  | 27858\.63  | 139329\.38 | 15527\.24  | 58422\.88  | 71173\.47  |
| DQN\_D  | 222784\.57 | 309020\.43 | 470983\.06 | 475030\.98 | 489336\.08 | 277699\.50 | 131327\.94 | 339454\.65 |
| DQN\_S  | 191688\.64 | 193456\.03 | 208328\.53 | 70652\.35  | 390723\.20 | 168427\.88 | 286287\.98 | 215652\.09 |
| PPO\_B  | 8222\.64   | 413922\.71 | 583\.47    | 572368\.55 | 251082\.20 | 52145\.48  | 46127\.77  | 192064\.69 |
| PPO\_D  | 103490\.23 | 145784\.87 | 425056\.06 | 146170\.73 | 123431\.65 | 39675\.94  | 244001\.22 | 175372\.96 |
| PPO\_S  | 213945\.06 | 112278\.37 | 14895\.42  | 88723\.09  | 162622\.98 | 130128\.74 | 365597\.64 | 155455\.90 |
| Average   | 166550\.36 | 177718\.45 | 145472\.96 | 170222\.05 | 192698\.59 | 96473\.90  | 163408\.84 |            |

![Score](Agent%20and%20reward%20comparison%20Experiment/plots/Mean_score.png?raw=true "Score")


### Score/Timesteps
| \{\}      | Bridge   | Overflow | Combined | Distance | Economic | Episode  | Redisp   | Average  |
|-----------|----------|----------|----------|----------|----------|----------|----------|----------|
| A2C\_B  | 1025\.05 | 1434\.72 | 735\.44  | 764\.68  | 898\.44  | 927\.30  | 1014\.40 | 971\.43  |
| A2C\_D  | 295\.76  | 806\.17  | 89\.75   | 29\.54   | 25\.16   | 1022\.26 | 39\.17   | 329\.69  |
| A2C\_S  | 1052\.26 | 38\.60   | 24\.81   | 38\.48   | 28\.28   | 38\.27   | 34\.09   | 179\.26  |
| DDPG\_B | 1214\.73 | 646\.78  | 1315\.35 | 755\.18  | 903\.27  | 1099\.66 | 901\.17  | 976\.59  |
| DQN\_D  | 737\.58  | 974\.37  | 1919\.87 | 851\.51  | 1125\.92 | 1350\.74 | 622\.64  | 1083\.23 |
| DQN\_S  | 623\.09  | 422\.99  | 537\.54  | 976\.40  | 1405\.02 | 435\.42  | 927\.37  | 761\.12  |
| PPO\_B  | 472\.84  | 1160\.68 | 280\.51  | 2662\.43 | 806\.61  | 624\.95  | 594\.97  | 943\.28  |
| PPO\_D  | 1026\.89 | 391\.15  | 1028\.54 | 372\.66  | 374\.19  | 34\.90   | 481\.58  | 529\.99  |
| PPO\_S  | 1482\.54 | 395\.17  | 597\.97  | 327\.60  | 487\.45  | 917\.63  | 479\.92  | 669\.75  |
| Average   | 881\.19  | 696\.74  | 725\.53  | 753\.16  | 672\.71  | 716\.79  | 566\.15  |          |

![Scoretimesteps](Agent%20and%20reward%20comparison%20Experiment/plots/Mean_scoretimesteps.png?raw=true "Scoretimesteps")

