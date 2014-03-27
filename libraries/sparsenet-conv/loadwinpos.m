
load win.mat

for w_idx = 1:length(win_num)
    sfigure(win_num(w_idx));
    set(win_num(w_idx), 'Position', win_pos(:,w_idx));
end

