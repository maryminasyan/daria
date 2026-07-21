import emcee
import numpy as np

class MoveAssembler(object):
    def __init__(self,move_types='default',**kwargs):
        '''
        move_types (str or list of str): emcee move(s) to use
        fractions (1 or list of float): associated fractions, if using
            several moves.
        '''
        self.move_types = move_types
        self.mix = isinstance(move_types,list)
        if self.mix:
            self.fractions = kwargs.pop('fractions',[1])
            assert np.sum(self.fractions) == 1, '`fractions` must sum to 1'
        
    def assemble_moves(self):
        move_types = self.move_types
        if move_types == 'default':
            return None
        elif not self.mix:
            return [self.get_move(move_types)]
        else:
            moves = []
            for i, move_type in enumerate(move_types):
                moves.append((self.get_move(move_type),self.fractions[i]))
            return moves

    def label(self):
        move_types = self.move_types
        if not self.mix:
            return move_types
        else:
            moves_label = ''
            for i,move_type in enumerate(move_types):
                moves_label += f'{move_type}{self.fractions[i]}_'
            moves_label += 'mix'
            return moves_label

    def get_move(self,move_type):
        if move_type.lower() == 'default':
            return None
        elif move_type.lower() == 'de':
            return emcee.moves.DEMove()
        elif move_type.lower() == 'desnooker':
            return emcee.moves.DESnookerMove()
        else:
            raise AssertionError('Unsupported move')
