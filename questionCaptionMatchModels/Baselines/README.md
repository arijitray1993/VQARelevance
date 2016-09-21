Pre-requisites:

Install Torch: 

```git clone https://github.com/torch/distro.git ~/torch --recursive```

```cd ~/torch; bash install-deps;```

``` ./install.sh ```

``` source ~/.bashrc ```

To run the baseline models, download the following data files (they are in torch format), and put them in this same folder:

All the files in https://filebox.ece.vt.edu/~ray93/qcbaselinedata/ . Just to list them out, they are:
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/dataset_small_lost_correspondence.t7
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/scores_allapplidata_dropout.t7
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/scores_raw_allapplidata.t7
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/dataset_small.t7
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/scores.json
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/scores_allapplidata_dropout2.t7
- https://filebox.ece.vt.edu/~ray93/qcbaselinedata/scores_raw_allapplidata2.t7
  
For Q-GEN Score:
- ``` python qgenscore.py ```

For VQA-MLP:
- ``` th rawscoreclassifier.py ```

For Entropy-based:
- ``` th trainmultiple_compute_entropies.lua ```

