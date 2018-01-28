Provenance of the data -- because we have TPM with filtering:

* Bladder

https://github.com/hammerlab/infino-rcc/blob/master/extract%20bladder%20counts%20and%20TPM%20from%20kallisto%20and%20filter%20genes.ipynb  is what is described below

extract bladder counts and TPM from kallisto and filter genes.ipynb: goes from kallisto output to tpm (called new.bladder.raw.tpm.csv)
then filter counts data for low gene counts
then choose those genes for tpm
that gives new.bladder.raw.tpm.FILTERED.tsv (which we then used in https://github.com/hammerlab/infino-rcc/blob/master/prepare%20bladder%20data%20for%20infino%20v2.ipynb ..)

new.bladder.raw.tpm.FILTERED.tsv is now here as bladder.tpm.tsv

* TCGA

same story
infino-rcc/tcga_kirc_data/tcga_kirc.1.subset.tpm.csv was before filtering for low read counts. (that was made in https://github.com/hammerlab/infino-rcc/blob/master/TCGA-KIRC%201%20Extract%20our%20cohort%20subset.ipynb)
then tcga_kirc.2.subset.tpm.filtered.tsv was after filtering. (that was made in https://github.com/hammerlab/infino-rcc/blob/master/TCGA-KIRC%202%20filter%20genes%20in%20raw%20counts%20and%20TPM.ipynb)

tcga_kirc.2.subset.tpm.filtered.tsv is now here as: tcgakirc.tpm.tsv


---

precise gsutil commands i ran to store in `~/Dropbox/tcgakirc_bladder_rawtpm`:

```
gsutil cp gs://hammerlab-infino-mz/homemaxim/immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/tcga_kirc_data/tcga_kirc.1.subset.tpm.csv .
gsutil cp gs://hammerlab-infino-mz/homemaxim/immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/tcga_kirc_data/tcga_kirc.2.subset.tpm.filtered.tsv .
gsutil cp gs://hammerlab-infino-mz/homemaxim/immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/new.bladder.raw.tpm.FILTERED.tsv .
gsutil cp gs://hammerlab-infino-mz/homemaxim/immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/new.bladder.raw.tpm.csv .
```

---

here's a version of how we could implement this in our new code to do the filtering 

```
data_tpm = read.csv('tcga_kirc_data/tcga_kirc.1.subset.tpm.csv', row.names=1)

head(data_tpm)

print(dim(data_tpm))

# clean it up -- remove rows with all 0s
clean_data_tpm = data_tpm[as.logical(rowSums(data_tpm) != 0),]
print(nrow(clean_data_tpm))

# make sure there are no NaNs
print("should be 0:")
print(sum(is.nan(as.matrix(clean_data_tpm))))

# global tpm range
tmp_flatten = as.matrix(clean_data_tpm)
dim(tmp_flatten) = NULL
summary(tmp_flatten)


remove low read count rows

as in "Run Immune Infiltrate Tools.ipynb": "Filter out genes where we have almost no data (i.e. top 10%)"

tmpvec = clean_data_counts %>%
        data.frame() %>%
        tibble::rownames_to_column("Gene_symbols") %>% # add index back as column
        mutate(low_cnt=rowSums(.[-1] < 1)) %>%
        dplyr::select(low_cnt)

print("discarding rows that have more than this number of 0s (in raw counts):")
quantile(tmpvec$low_cnt, .90)

# this makes sense as a threshold for counts
low_expression_threshold = 1
clean_data_counts_filtered <- clean_data_counts %>%
        data.frame() %>%
        tibble::rownames_to_column("Gene_symbols") %>% # add index back as column
        mutate(low_cnt=rowSums(.[-1] < low_expression_threshold)) %>% # sum all columns except gene symbols
        filter(low_cnt < quantile(low_cnt, .90)) %>%  # FIXME: Changed from `<` to `<=`...didn't work when low_cnt was 0 for most/all.
        dplyr::select(-low_cnt) %>%
        write_tsv("tcga_kirc_data/tcga_kirc.2.subset.counts.filtered.tsv") %>%
        tibble::column_to_rownames("Gene_symbols")

nrow(clean_data_counts_filtered)
row.names(head(clean_data_counts_filtered))
```

that's raw counts
for tpm:
tpm has different range, so maybe just rowSums rather than rowSums under a threshold
and do quantile .10 instead of .90