import numpy as np


class Moffat2D:
    """
    Moffat 2D generator
    """

    def __init__(self, cutout_size=21, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.train_history = None
        self.cutout_size = cutout_size
        self.x, self.y = np.indices((cutout_size, cutout_size))

    def moffat2D_model(self, a, x0, y0, sx, sy, theta, b, beta):
        # https://pixinsight.com/doc/tools/DynamicPSF/DynamicPSF.html
        dx_ = self.x - x0
        dy_ = self.y - y0
        dx = dx_ * np.cos(theta) + dy_ * np.sin(theta)
        dy = -dx_ * np.sin(theta) + dy_ * np.cos(theta)

        return b + a / np.power(1 + (dx / sx) ** 2 + (dy / sy) ** 2, beta)

    def sigma_to_fwhm(self, beta):
        return 2 * np.sqrt(np.power(2, 1 / beta) - 1)

    def random_model_label(self, N=10000, flatten=False, progress=False, return_all=False):

        images = []
        labels = []
        params = dict(a=[], x0=[], y0=[], theta=[], b=[], beta=[], sx=[], sy=[], noise=[])

        if not progress:
            def progress(x): return x
        else:
            def progress(x): return tqdm(x)
            
        a = np.ones(N)
        b = np.zeros(N)
        x0, y0 = np.random.uniform(3, self.cutout_size - 3, (2, N))
        theta = np.random.uniform(0, np.pi / 8, size=N)
        beta = np.random.uniform(1, 8, size=N)
        sx = np.array([np.random.uniform(2.5, 7.5) / self.sigma_to_fwhm(_beta) for _beta in beta])
        sy = np.random.uniform(0.5, 1.5, size=N) * sx
        noise = np.random.uniform(0, 0.1, size=N)
        
        for i in progress(range(N)):          
            _noise = np.random.rand(self.cutout_size, self.cutout_size)*noise[i]
            data = self.moffat2D_model(a[i], x0[i], y0[i], sx[i], sy[i], theta[i], b[i], beta[i]) + _noise
        
            images.append(data)
        
        images = np.array(images)[:, :, :, None]
        
        if not return_all:
            labels = np.array([x0, y0]).T
        else:
            labels = np.array([a, x0, y0, sx, sy, theta, b, beta, noise]).T

        if N == 1 and flatten:
            return (np.array(images[0]), np.array(labels[0]))
        else:
            return (np.array(images), np.array(labels))