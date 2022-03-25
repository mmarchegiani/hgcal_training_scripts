from colour import Color
import numpy as np
import evaluation as ev
from colorwheel import ColorWheel, HighlightColorwheel, ColorwheelWithProps

def safe_divide(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def get_stats(tbeta=.2, td=.5, nmax=4):
    yielder = ev.TestYielder()
    matching_stats = ev.Stats()
    event_stats = ev.Stats()
    all_matches = []
    yielder.reset_loader()
    n_done = 0
    for event, _, clustering, matches in yielder.iter_matches(tbeta, td, nmax):
        matching_stats.extend(ev.statistics_per_match(event, clustering, matches))
        all_matches.extend(matches)
        event_stats.add('confmat', ev.signal_to_noise_confusion_matrix(event, clustering, norm=True))
        event_stats.extend(ev.get_matched_vs_unmatched(event, clustering, matches))
        n_done += 1

        # Some basic stats per category
        splitting_per_category = np.zeros((4, 3))
        for truth_ids, pred_ids in matches:
            cat = ev.get_category([event.truth_pdgid_by_id(id) for id in truth_ids])
            n_pred = len(pred_ids)
            n_truth = len(truth_ids)
            if n_pred == 1 and n_truth == 1: # One-to-one
                splitting_per_category[cat, 0] += 1
            elif n_pred > 1 and n_truth == 1: # Oversplit
                splitting_per_category[cat, 1] += 1
            elif n_pred == 1 and n_truth > 1: # Overmerged
                splitting_per_category[cat, 2] += 1
            else:
                raise Exception('Impossible')
        n_per_cat = splitting_per_category.sum(axis=1)
        event_stats.add('matches_per_cat', n_per_cat)
        event_stats.add('nmatchsplittype_per_cat', splitting_per_category)

    return matching_stats, all_matches, event_stats


def print_pdgids():
    yielder = ev.TestYielder()
    for event in yielder.iter(400):
        print(np.unique(event.truth_pdgid))


def test():
    matching_stats, matches, event_stats = get_stats(nmax=300)

    print(
        'Avg fraction unmatched showers pred:',
        (event_stats['n_showers_unmatched_pred']/event_stats['n_showers_pred']).mean()
        )
    print(
        'Avg fraction unmatched showers truth:',
        (event_stats['n_showers_unmatched_truth']/event_stats['n_showers_truth']).mean()
        )

    for i_cat, cat in enumerate(['em', 'had', 'mip']):
        print(
            f'[{cat}] Avg fraction unmatched showers truth:',
            (event_stats[f'n_showers_unmatched_truth_{cat}']/event_stats[f'n_showers_truth_{cat}']).mean()
            )

    print('Fraction of match type per category:')
    print(event_stats['matches_per_cat'].sum(axis=0) / event_stats['matches_per_cat'].sum())

    print('Splitting type per category:')
    fracmatchsplittype_per_cat = event_stats['nmatchsplittype_per_cat'].sum(axis=0)
    fracmatchsplittype_per_cat /= np.expand_dims(fracmatchsplittype_per_cat.sum(axis=-1), -1)
    print(fracmatchsplittype_per_cat)
    
    confmat_nhits = event_stats['confmat'][:,0,:,:].mean(axis=0)
    confmat_edep = event_stats['confmat'][:,1,:,:].mean(axis=0)
    print('confmat_nhits:')
    print(confmat_nhits)
    print('confmat_edep:')
    print(confmat_edep)

    print('fraction_hitenergy_matched_truth:', event_stats['fraction_hitenergy_matched_truth'].mean())
    print('fraction_hitenergy_matched_pred:', event_stats['fraction_hitenergy_matched_pred'].mean())
    print('fraction_hitenergy_unmatched_truth:', event_stats['fraction_hitenergy_unmatched_truth'].mean())
    print('fraction_hitenergy_unmatched_pred:', event_stats['fraction_hitenergy_unmatched_pred'].mean())

    print('fraction_nhits_matched_truth:', event_stats['fraction_nhits_matched_truth'].mean())
    print('fraction_nhits_matched_pred:', event_stats['fraction_nhits_matched_pred'].mean())
    print('fraction_nhits_unmatched_truth:', event_stats['fraction_nhits_unmatched_truth'].mean())
    print('fraction_nhits_unmatched_pred:', event_stats['fraction_nhits_unmatched_pred'].mean())



def look_for_weird_matches():
    tbeta = .2
    td = .5

    # outfile = ev._make_parent_dirs_and_format('plots_matches_%b%d/overmerged.html', touch=True)
    # def find_overmerged_match(event, clustering, truth_ids, pred_ids):
    #     sel_truth = np.isin(event.y, truth_ids)
    #     sel_pred = np.isin(clustering, pred_ids)
    #     return event.energy[sel_pred].sum() > 2.*event.energy[sel_truth].sum()
    # weird_matches_to_file(outfile, find_overmerged_match, tbeta, td, n=10)

    outfile = ev._make_parent_dirs_and_format('plots_matches_%b%d/undercovered.html', touch=True)
    def find_undercovered_match(event, clustering, truth_ids, pred_ids):
        sel_truth = np.isin(event.y, truth_ids)
        sel_pred = np.isin(clustering, pred_ids)
        return event.energy[sel_pred].sum() < .5*event.energy[sel_truth].sum()
    weird_matches_to_file(outfile, find_undercovered_match, tbeta, td, n=5)


def weird_matches_to_file(outfile, match_selector, tbeta, td, yielder=None, nmax=None, n=None):
    if yielder is None: yielder = ev.TestYielder()
    n_found = 0
    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta, td, nmax=None)):
        npz = yielder.dataset.npzs[event.inpz]
        found_weird_match = False
        colorwheel = HighlightColorwheel()
        colorwheel.assign(-1, '#bfbfbf')
        colorwheel.assign(0, '#bfbfbf')
        for truth_ids, pred_ids in matches:
            all_ids = np.concatenate((truth_ids, pred_ids))
            if match_selector(event, clustering, truth_ids, pred_ids):
                colorwheel.highlight_many(all_ids)
                if not found_weird_match: n_found += 1
                found_weird_match = True
            else:
                colorwheel.normal_many(all_ids)
        if not found_weird_match:
            print(f'Not saving event {i} ({n_found=}/{n})')
            continue
        print(f'Saving event {i} ({n_found=}/{n}) ({npz})')
        pdata_pred = ev.compile_plotly_data(event, clustering, colorwheel)
        pdata_truth = ev.compile_plotly_data(event, event.y, colorwheel)
        with open(outfile, 'a') as f:
            f.write(f'\n<h2>Event {i}; {npz}</h2>\n')
        ev.side_by_side_pdata_to_file(
            outfile, pdata_pred, pdata_truth,
            title1='Predicted', title2='Truth',
            mode='a'
            )
        if n is not None and n == n_found:
            print(f'Found {n} examples; quiting')
            return

def matched_colorwheel(matches, **kwargs):
    """
    One color is one match, also for over/undersplitting
    """
    colorwheel = ColorWheel(**kwargs)
    colorwheel.many([0, -1], '#bfbfbf')
    for truth_ids, pred_ids in matches:
        colorwheel.many(np.concatenate((truth_ids, pred_ids)))
    return colorwheel

def matched_colorwheel_only_primary(matches, **kwargs):
    """
    Only the 'main' match (largest IoM) is given the same color; other
    matches are given a different color.
    """
    colorwheel = ColorWheel(**kwargs)
    colorwheel.many([0, -1], '#bfbfbf')
    for truth_ids, pred_ids in matches:
        main_match = [truth_ids[0], pred_ids[0]]
        others = list(truth_ids[1:]) + list(pred_ids[1:])
        colorwheel.many(main_match)
        (colorwheel(other) for other in others)
    return colorwheel

def matched_colorwheel_with_props(matches, **kwargs):
    """
    The main match is given the color at 100% opacity, and subsequent matches
    are the same color but with an alpha.
    """
    colorwheel = ColorwheelWithProps(**kwargs)
    colorwheel.many([0, -1], color='#bfbfbf', alpha=.6)
    for truth_ids, pred_ids in matches:
        # Reshuffle so the main match is first
        ids = [ truth_ids[0], pred_ids[0] ] + list(truth_ids[1:]) + list(pred_ids[1:])
        if len(ids) > 2:
            print(f'Main match: {truth_ids[0]} to {pred_ids[0]}; alpha for clusters {ids[2:]}')
        alpha = [.3 for i in range(len(ids))]
        alpha[0] = 1.
        alpha[1] = 1.
        colorwheel.many(ids, alpha=alpha)
    return colorwheel


def test_colorwheel_with_props():
    cwp = ColorwheelWithProps()

    cwp.assign(1, alpha=.2, size=4)

    cwp.many([2,3,4], alpha=.2, size=[5,6,7])

    print(cwp(2))
    print(cwp(3))
    print(cwp(4))






def specific_cat_only_plots():
    tbeta = .2
    td = .5
    outfile = ev._make_parent_dirs_and_format('plots_evaluation_%b%d/em_only.html', touch=True)

    yielders = {
        'em' : ev.TestYielderEM(),
        'had' : ev.TestYielderHAD(),
        'mip' : ev.TestYielderMIP(),
        }

    for cat, yielder in yielders.items():
        print(f'Making plots for {cat}')
        n_done = 0
        for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta, td, nmax=None)):
            if n_done == 20:
                break
            elif n_done % 5 == 0:
                print(f'Done {n_done}/20')
                outfile = ev._make_parent_dirs_and_format(f'plots_evaluation_%b%d_100/{cat}_{n_done}.html', touch=True)

            npz = yielder.dataset.npzs[event.inpz]
            colorwheel = matched_colorwheel(matches)
            pdata_pred = ev.compile_plotly_data(event, clustering, colorwheel)
            pdata_truth = ev.compile_plotly_data(event, event.y, colorwheel)
            with open(outfile, 'a') as f:
                f.write(f'\n<h2>Event {i}; {npz}</h2>\n')
            ev.side_by_side_pdata_to_file(
                outfile, pdata_pred, pdata_truth,
                title1='Predicted', title2='Truth',
                mode='a'
                )
            n_done += 1


def test_single_photon():
    tbeta = .2
    td = .5
    nmax = 1
    outfile = ev._make_parent_dirs_and_format('test.html', touch=True)
    yielder = ev.TestYielderSinglePhoton()
    # yielder.model.signal_threshold = .5
    for i, (event, prediction, clustering) in enumerate(yielder.iter_clustering(tbeta, td, nmax=nmax)):
        pdata_pred = ev.compile_plotly_data(event, clustering)
        pdata_truth = ev.compile_plotly_data(event, event.y)
        pdata_pred_clusterspace = ev.compile_plotly_data_clusterspace(event, prediction, clustering)
        pdata_truth_clusterspace = ev.compile_plotly_data_clusterspace(event, prediction, event.y)

        ev.side_by_side_pdata_to_file(
            outfile, pdata_pred, pdata_truth,
            title1='Predicted', title2='Truth',
            mode='a'
            )
        ev.side_by_side_pdata_to_file(
            outfile, pdata_pred_clusterspace, pdata_truth_clusterspace,
            title1='Predicted', title2='Truth',
            mode='a'
            )

def regular_plots():
    import plotly.graph_objects as go
    tbeta = .2
    td = .5
    yielder = ev.TestYielder()
    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta, td, nmax=4)):
        if i < 3: continue
        npz = yielder.dataset.npzs[event.inpz]
        colorwheel = matched_colorwheel(matches)
        # colorwheel = matched_colorwheel(matches, colors='viridis')
        # colorwheel = matched_colorwheel_with_props(matches, colors='viridis')
        # colorwheel = matched_colorwheel_only_primary(matches, colors='viridis') # <-- ACAT!
        pdata_pred = ev.compile_plotly_data(event, clustering, colorwheel)
        pdata_truth = ev.compile_plotly_data(event, event.y, colorwheel)
        outfile = ev._make_parent_dirs_and_format(f'plots_evaluation_%b%d/{i:03d}.html', touch=True)
        with open(outfile, 'a') as f:
            f.write(f'\n<h2>Event {i}; {npz}</h2>\n')

        ev.side_by_side_pdata_to_file(
            outfile, pdata_pred, pdata_truth,
            title1='Predicted', title2='Truth',
            mode='a',
            # width=1000,
            legend=False
            )

        # ev.single_pdata_to_file(
        #     'test.html', pdata_pred,
        #     title='Predicted',
        #     mode='w',
        #     width=800
        #     )

        # width = 1000
        # fig = go.Figure(data=pdata_pred, layout_title_text='Prediction')
        # scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)', zaxis_title='y (cm)', aspectmode='cube')
        # fig.update_layout(width=width, height=width, scene=scene)
        # fig.write_image("test.png")

        if i==3: break


def test_cpu_inference_speed():
    yielder = ev.TestYielderTestYielder()
    import datetime
    start = datetime.datetime.now()
    n = 100
    for i, *_ in enumerate(yielder.iter_pred(nmax=n)):
        continue
    end = datetime.datetime.now()
    n_seconds = (end-start).total_seconds()
    print(f'Total time: {n_seconds}')
    r = n_seconds/n
    print(f'avg time: {r} s / inf')
    print(f'          {1./r} inf / s')


def plot_noise_distributions():
    import matplotlib.pyplot as plt
    tbeta = .2
    td = .5
    nmax = 1
    tau_yielder = ev.TestYielder().iter(nmax=nmax)
    photon_yielder = ev.TestYielderSinglePhoton().iter(nmax=nmax)

    tau_event = next(tau_yielder)
    tau_event = tau_event[tau_event.select_noise_hits]
    tau_event.name = 'taus'
    photon_event = next(photon_yielder)
    photon_event = photon_event[photon_event.select_noise_hits]
    photon_event.name = 'photons'

    feature_titles = {
        'energy'      : 'E (MeV)',
        'etahit'      : r'$\eta$',
        'zerofeature' : 'Zero',
        'thetahit'    : r'$\theta$',
        'rhit'        : 'R (cm)',
        'xhit'        : 'x (cm)',
        'yhit'        : 'y (cm)',
        'zhit'        : 'z (cm)',
        'time'        : 't (?)'
        }

    for feature in ['energy', 'etahit', 'zerofeature', 'thetahit', 'rhit', 'xhit', 'yhit', 'zhit', 'time']:
        outfile = ev._make_parent_dirs_and_format(f'plots_noisedist_%b%d/{feature}.png')
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()

        _, bins, _ = ax.hist(getattr(tau_event, feature), bins=100, label='taus', density=True)
        ax.hist(getattr(photon_event, feature), bins=bins, label='photons', density=True, alpha=.6)

        ax.set_xlabel(feature_titles[feature])
        ax.legend()
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # print_pdgids()
    # test()
    # look_for_weird_matches()
    # specific_cat_only_plots()
    test_single_photon()
    # regular_plots()
    # test_cpu_inference_speed()
    # test_colorwheel_with_props()

    # plot_noise_distributions()