from stancache.stancache import cached, cached_stan_fit, _read_file
import stancache

stancache.config.set_value(cache_dir = '/home/jovyan/modelcache/mz')
stancache.seed.set_seed()

