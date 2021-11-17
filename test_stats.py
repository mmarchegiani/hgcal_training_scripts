import numpy as np
import evaluation as ev
from colorwheel import ColorWheel, HighlightColorwheel

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
            cat = ev.get_category(truth_ids)
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



def test():
    matching_stats, matches, event_stats = get_stats(nmax=40)

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

def matched_colorwheel(matches):
    colorwheel = ColorWheel()
    colorwheel.many([0, -1], '#bfbfbf')
    for truth_ids, pred_ids in matches:
        colorwheel.many(np.concatenate((truth_ids, pred_ids)))
    return colorwheel


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


if __name__ == '__main__':
    # test()
    # look_for_weird_matches()
    specific_cat_only_plots()