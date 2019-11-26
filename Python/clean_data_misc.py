
class wannier_et_al:

    def __init__(self, path):
        self.path = path


    def get_gene_lengths2(self):
        gene_sites = self.path + '/data/Wannier_et_al/gene_sites.txt'
        with open(gene_sites) as gs:
             rows = ( line.strip().split('\t') for line in gs )
             gs_dict = { row[1]:row[0] for row in rows if row[1] != 'Position'}

        genome_path = self.path + '/data/Wannier_et_al/CP025268.1.txt'
        fasta = pt.classFASTA(genome_path).readFASTA()
        locus_tags = []
        protein_ids = []
        starts = []
        stops = []
        lengths = []
        gene_names = []
        num_good_list = []
        for gene in fasta:
            lengths.append(len(gene[1]))
            header = gene[0].split(' ')
            locus_tag = re.findall(r"[\w']+", [s for s in header if 'locus_tag' in s][0])[1]
            locus_tags.append(locus_tag)
            location = re.findall(r"[\w']+", [s for s in header if 'location=' in s][0])
            start = location[-2]
            starts.append(start)
            stop = location[-1]
            stops.append(stop)

            number_good = 0
            gs_genessss= []
            for gs_site, gs_gene in gs_dict.items():
                if (int(start) <= int(gs_site)) and (int(stop) >= int(gs_site)):
                    number_good += 1
                    gs_genessss.append(gs_gene)
            #num_good_list.append(number_good)

            if number_good > 1:
                print( locus_tag, gs_genessss)

    def get_gene_lengths(self):
        #genome_path = self.path + '/data/Wannier_et_al/NC_000913.3.txt'
        #fasta = pt.classFASTA(genome_path).readFASTA()
        #gene_size_dict = {}
        #for gene in fasta:
        #    header = gene[0].split(' ')
        #    gene_name = [s for s in header if 'gene=' in s][0].split('=')[1].split(']')[0]
        #    gene_len = len(gene[1])
        #    gene_size_dict[gene_name] = gene_len
        #gene_size_dict_out = mydir + 'data/Wannier_et_al/gene_size_dict.txt'
        #with open(gene_size_dict_out, 'wb') as handle:
        #    pickle.dump(gene_size_dict, handle)
        gene_size_dict = {}
        to_ignore = ['insN', 'crl', 'yaiX', 'yaiT', 'renD', 'nmpC', 'lomR', \
                    'ydbA', 'ydfJ', 'yoeA', 'wbbL', 'gatR', 'yejO', 'yqiG', \
                    'yhcE', 'yrhA', 'yhiS', 'insO']
        with open(self.path + '/data/Wannier_et_al/NC_000913_3.gb', "rU") as input_handle:
            for record in SeqIO.parse(input_handle, "genbank"):
                for feature in record.features:
                    if feature.type == 'gene':
                        gene_name = feature.qualifiers['gene'][0]
                        if gene_name in to_ignore:
                            continue
                        start_stop = (re.findall(r"[\w']+", str(feature.location)))
                        gene_len = int(start_stop[1]) - int(start_stop[0])
                        gene_size_dict[gene_name] = gene_len

        gene_size_dict_out = mydir + 'data/Wannier_et_al/gene_size_dict.txt'
        with open(gene_size_dict_out, 'wb') as handle:
            pickle.dump(gene_size_dict, handle)

    def clean_data(self):
        # ALEdb setup
        # A = ALE
        # F = Flask
        # I = isolate number
        # R = Technical replicate
        treats = ['C321', 'C321.deltaA', 'C321.deltaA.earlyfix', 'ECNR2.1']
        #treats = ['C321.deltaA.earlyfix']
        table = str.maketrans(dict.fromkeys('""'))
        #gene_site_dict = {}
        for treat in treats:
            gene_treat_dict = {}
            pop_position_dict = {}
            lengths = []
            treat_path = self.path + '/data/Wannier_et_al/' + treat + '_mutation_table.txt'
            for i, line in enumerate(open(treat_path, 'r')):
                line = line.strip('\n')
                items = line.split("\t")
                items = [item.translate(table) for item in items]
                if i == 0:
                    samples = [x.split('>')[1][:-3] for x in items[5:]]
                    samples = [x.replace(' ', '-') for x in samples]
                    if (treat == 'C321.deltaA') or (treat == 'C321.deltaA.earlyfix'):
                        samples_list = [list(x) for x in samples]
                        for x in samples_list:
                            x[5] = 'delta'
                        samples = [''.join(x) for x in samples_list]
                    samples_no_FIR = [s.rsplit('-', 3)[0] for s in samples]
                    for sample_no_FIR in list(set(samples_no_FIR)):
                        gene_treat_dict[sample_no_FIR] = {}
                    for j, sample_no_FIR in enumerate(samples_no_FIR):
                        pop_position_dict[j] = sample_no_FIR

                # remove noncoding and mutations in overlaping genes
                if ('noncoding' in items[4]) or ('pseudogene' in items[4]) \
                        or ('intergenic' in items[4]) or (';' in items[3]) \
                        or (len(set(items[5:])) == 1):
                    continue

                just_muts = items[5:]
                gene = items[3]
                if len(gene.split(',')) > 1:
                    continue
                #gene_site_dict[gene] = items[0]
                # merge reps
                site_dict = {}
                for m, mut_m in enumerate(just_muts):
                    if mut_m == '1':
                        site_dict[pop_position_dict[m]] = 1

                if len(site_dict) > 3:
                    continue
                for key, value in site_dict.items():
                    if gene in gene_treat_dict[key]:
                        gene_treat_dict[key][gene] += 1
                    else:
                        gene_treat_dict[key][gene] = 1

            df = pd.DataFrame.from_dict(gene_treat_dict).T
            df = df.fillna(0)
            df_out = self.path + '/data/Wannier_et_al/' + treat + '_mutation_table_clean.txt'
            df.to_csv(df_out, sep = '\t', index = True)


        #gene_site_dict_out = open(self.path + '/data/Wannier_et_al/gene_sites.txt', 'w')
        #gene_site_dict_out.write('\t'.join(['gene', 'site']) + '\n')
        #for gene_i, site_i in gene_site_dict.items():
        #    gene_site_dict_out.write('\t'.join([gene_i, site_i.replace(',', '')]) + '\n')
        #gene_site_dict_out.close()
