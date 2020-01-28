

class kryazhimskiy_et_al:

    def clean_table_s5(self):
        in_path = mydir + 'data/Kryazhimskiy_et_al/NIHMS658386-supplement-Table_S5.txt'
        df_out = open(mydir + 'data/Kryazhimskiy_et_al/table_S5_clean.txt', 'w')
        column_headers = ['Founder', 'Pop', 'Clone', 'Notes', 'Chr', 'Pos', \
                        'Unique_Mutation_ID', 'Ancestral_allele', 'Mutant_allele', \
                        'Is_convergent', 'Type', 'Gene', 'AA_change', 'AA_position', \
                        'Distance_to_gene', 'COV_IN_CLONE', 'CNT_IN_CLONE', \
                        'FREQ_IN_CL', 'FREQ_OUT_CL', 'Posterior_Probability', 'Notes']
        df_out.write('\t'.join(column_headers) + '\n')
        for line in open(in_path):
            line_split = line.split('\t')[:-11]
            df_out.write('\t'.join(line_split) + '\n')
        df_out.close()


    #def pop_by_gene_kryazhimskiy(self):
    #    df_path = mydir + 'data/Kryazhimskiy_et_al/NIHMS658386-supplement-Table_S5.txt'
    #    df = pd.read_csv(df_path, sep = '\t', header = 'infer', index_col = 0)
    #
    #    print(df.columns)

    def get_size_dict(self):
        in_path = mydir + 'data/Kryazhimskiy_et_al/Saccharomyces_cerevisiae_W303_Greg_Lang/w303_ref.gff'
        df_out = open(mydir + 'data/Kryazhimskiy_et_al/Saccharomyces_cerevisiae_W303_Greg_Lang/w303_ref_clean.gff', 'w')
        df_out.write('\t'.join(['Gene', 'ID', 'Parents', 'Length']) + '\n')
        for line in open(in_path):
            line_split = line.split()
            if (len(line_split) < 3) or (line_split[2] != 'CDS'):
                continue
            genes = [x for x in line_split[-1].split(';') if 'gene=' in x]
            if len(genes) == 1:
                gene = str(genes[0].split('=')[1])
            else:
                gene = ''
            ids= [x for x in line_split[-1].split(';') if 'ID=' in x]
            if len(ids) == 1:
                id = str(ids[0].split('=')[1])
            else:
                id = ''
            parents = [x for x in line_split[-1].split(';') if 'Parent=' in x]
            if len(parents) == 1:
                parent = str(parents[0].split('=')[1])
            else:
                parent = ''
            length = str(int(line_split[4]) - int(line_split[3]))
            df_out.write('\t'.join([gene, id, parent, length]) + '\n')
        df_out.close()
