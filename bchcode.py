# Based on: https://github.com/tomekrzymyszkiewicz/forward-error-correction.git
import math
import multiprocessing
import os

import komm as komm
import numpy as np
from tqdm import tqdm

from utils import logger

logger = logger.get_logger(__name__)


class BCHCode:

    def __init__(self, binary_data):
        self.binary_data = np.array(binary_data)

    def __calculate_zeros_addition_bch(self, part_length):
        bits = self.binary_data
        additional_zeros = (math.ceil(len(bits) / part_length) * part_length) - len(bits)
        return additional_zeros

    def encode(self, parameter, correcting_capability):
        """
        BCH code encoding method
        1 <= correcting_capability < 2^(parameter -1)
        """

        bits = self.binary_data
        code = komm.BCHCode(parameter, correcting_capability)

        bits = np.append(bits, [
            np.zeros(self.__calculate_zeros_addition_bch(code.dimension),
                     dtype=np.uint8)])
        parts_to_encode = np.reshape(bits, (-1, code.dimension), order='C')

        pool = multiprocessing.Pool(os.cpu_count())
        encoded_parts = list(
            tqdm(pool.imap(code.encode, parts_to_encode, chunksize=10),
                 total=parts_to_encode.shape[0],
                 leave=False,
                 desc="Encoding binary data with BCH"))

        encoded_parts = np.array(encoded_parts)

        logger.info(f"BCH statistics: dimension={code.dimension}, length={code.length}, size={encoded_parts.size}")

        return list(np.concatenate(encoded_parts))

    def decode(self, parameter, correcting_capability, message_shape):
        """
        Decoding method for cyclic BCH code
        """
        bits = self.binary_data
        code = komm.BCHCode(parameter, correcting_capability)
        max_parts = math.ceil(np.prod(message_shape) / code.dimension)

        logger.info(
            f"BCH statistics: dimension={code.dimension}, length={code.length}, size_to_decode={max_parts * code.length}")

        bits = np.append(bits, [
            np.zeros(self.__calculate_zeros_addition_bch(code.length),
                     dtype=np.uint8)])
        parts_to_decode = np.reshape(bits, (-1, code.length), order='C')[:max_parts]

        pool = multiprocessing.Pool(os.cpu_count())
        decoded_parts = list(
            tqdm(pool.imap(code.decode, parts_to_decode, chunksize=10),
                 total=parts_to_decode.shape[0],
                 leave=False,
                 desc="Decoding binary data with BCH"))

        decoded_parts = np.array(decoded_parts)
        decoded_parts = np.concatenate(decoded_parts)

        if (len(self.binary_data) % code.dimension) != 0:
            for i in range(0, self.__calculate_zeros_addition_bch(code.dimension)):
                decoded_parts = np.delete(decoded_parts, len(decoded_parts) - 1)

        return list(decoded_parts)
