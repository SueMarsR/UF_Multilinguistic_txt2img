export checkpoint_name="sd_diffuser_gpt_test_ZH_5e-5"
for i in {10..90..5}; do
    rm -rf "./work_dir/$checkpoint_name/pipeline-$i"
done


