
# Ideally cell features will someday be part of rinfino
# At that point we'll put this code in a merge method for cell features, just like we merge expression and sampleinfo today
# But for now we can combine like this
# Run with: docker run -it --rm -v $(pwd):/src hammerlab/infino_env:latest sh -c "cd /src && Rscript combine_cell_features.R"

library(dplyr)

# preserves column names (don't map - to .)
df1 = read.csv('../data/singleorigin.cellfeatures.tsv', sep='\t', check.names=F)
df2 = read.csv('../data/rcctils.cellfeatures.tsv', sep='\t', check.names=F)

# make sure first column (celltype) is named the same thing
names(df1)[1] = 'celltype'
names(df2)[1] = 'celltype'

# combine
combined = dplyr::bind_rows(df1, df2)
# fill na
combined[is.na(combined)] = 0

combined

write.table(combined, "out/singleorigin_plus_rcctils.combined.cellfeatures.tsv", sep='\t', row.names=F) # don't include index