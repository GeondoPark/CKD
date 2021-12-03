import os
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

task_names = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst-2", "sts-b"]
testtextfile = {"cola":"CoLA.tsv", 
                "mnli":["MNLI-m.tsv", "MNLI-mm.tsv"], 
                "mrpc":"MRPC.tsv", 
                "qnli":"QNLI.tsv",
                "qqp":"QQP.tsv",
                "rte":"RTE.tsv",
                "sst-2":"SST-2.tsv",
                "sts-b":"STS-B.tsv",
                }

exp_name = "model_name"

outdir = "./testdir/{}".format(exp_name)
if os.path.isdir(outdir):
    pass
else:
    os.mkdir(outdir)
##Since we do not conduct on this experimnets, we use the default results. (Refer GLUE BENCHMARK website)
os.system("cp ./testdir/AX.tsv {}".format(outdir))
os.system("cp ./testdir/WNLI.tsv {}".format(outdir))
for task in task_names:
    modeldir = "./experiments/exp_distil"+"_{}".format(task)+"/{}/checkpoints/best_checkpoint".format(exp_name, seed)

    script = "CUDA_VISIBLE_DEVICES={} python main_glue.py \
            --exp_name test \
            --do_test \
            --test_output_dir {} \
            --model_type bert \
            --model_path {} \
            --do_lower_case \
            --task_name {} \
            --per_gpu_batch_size 512".format(args.gpu, outdir, modeldir, task)
    
    os.system(script)
    if task == "mnli":
        script2 = "mv {}/test_results_{}.txt {}/{}".format(outdir, task, outdir, testtextfile[task][0])
        script3 = "mv {}/test_results_{}-mm.txt {}/{}".format(outdir, task, outdir, testtextfile[task][1])
        os.system(script2)
        os.system(script3)

    else:
        script2 = "mv {}/test_results_{}.txt {}/{}".format(outdir, task, outdir, testtextfile[task])
        os.system(script2)
zipscript = "zip {}/test.zip {}/*.tsv".format(outdir, outdir)
os.system(zipscript)


    
