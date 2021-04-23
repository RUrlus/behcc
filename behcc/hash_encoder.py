from behcc_extension import HashBinaryEncoder as Base

class HashBinaryEncoder(Base):
    """ Encodes the bits of the hashed inputs as an integer array.

    The eucledian distance between any encoding vectors
    follows an approximate Normal distribution.
    """

    def fit(self, X=None, y=None):
        """Fit encoding.

        Parameters
        ----------
        X : np.ndarray, optional
            numpy array to be hashed and encoded
        y : np.ndarray, optional
            param is ignored and only exists for
            compatibility with scikit-pipeline

        """
        super().fit(X, y)
        params = self.get_params()
        self.encoding_size_ = params['encoding_size']
        self.seed_ = params['seed']
        return self

    def transform(self, X):
        """Create binary encoding of hash of the input.

        Parameters
        ----------
        X : np.ndarray
            1D numpy array to be hashed and encoded

        Returns
        -------
        np.ndarray[int32]
            array containing the encoding

        """
        return super().transform(X)

    def fit_transform(self, X, y=None):
        """Create binary encoding of hash of the input.

        Parameters
        ----------
        X : np.ndarray
            1D numpy array to be hashed and encoded
        y : np.ndarray, optional
            param is ignored and only exists for
            compatibility with scikit-pipeline

        Returns
        -------
        np.ndarray[int32]
            array containing the encoding

        """
        super().fit(X, y)
        params = self.get_params()
        self.encoding_size_ = params['encoding_size']
        self.seed_ = params['seed']
        return super().transform(X)

    def _more_tags(self):
        return {
            'requires_fit': False,
            'allow_nan': False,
            'stateless': True,
            'preserves_dtype': [],
        }
