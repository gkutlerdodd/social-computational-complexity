# -*- coding: utf-8 -*-

from OutputWriter import OutputWriter


class MetapopulationOutput(OutputWriter):

    def __init__(self, metapopulation, filename='metapopulation.csv',
                 header=True, include_uuid=False, compress=False):
        fieldnames = ['Time', 'Births', 'Size','AverageFitness', 'CooperatorProportion', 'MaxCooperatorFitness', 'MaxDefectorFitness', 'AverageDeltaFitness']
        super(MetapopulationOutput, self).__init__(metapopulation=metapopulation,
                                                   filename=filename,
                                                   fieldnames=fieldnames,
                                                   header=header,
                                                   include_uuid=include_uuid,
                                                   compress=compress)


    def update(self, time):
        fits = self.metapopulation.max_fitnesses()
        record = {'Time': time,
                  'Births': self.metapopulation.num_births,
                  'Size': self.metapopulation.size(),
                  'AverageFitness':self.metapopulation.average_fitness(),#SV Check
                  'CooperatorProportion': self.metapopulation.prop_producers(),
                  'MaxCooperatorFitness': max(fits[0]),
                  'MaxDefectorFitness': max(fits[1]),
                  'AverageDeltaFitness': self.metapopulation.average_delta_fitness()}
        self.writerow(record)

