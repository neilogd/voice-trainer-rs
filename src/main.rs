use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    error::Error,
    io,
    time::{Duration, Instant},
};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType},
    Frame, Terminal,
};

extern crate portaudio;
use portaudio as pa;

extern crate ringbuf;

use std::sync::Arc;
use rustfft::{Fft, FftPlanner, num_complex::Complex};

const SAMPLE_RATE: f64 = 44_100.0;
const FRAMES: usize = 2048;           
const RINGBUF_SIZE: usize = FRAMES * 8;
const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true;

pub type AudioRingBuffer = ringbuf::HeapRb::<f32>;
type AudioConsumer = ringbuf::Consumer<f32, Arc<AudioRingBuffer>>;
type AudioProducer = ringbuf::Producer<f32, Arc<AudioRingBuffer>>;

struct AudioBackend {
    

    pa: pa::PortAudio,
    stream: Option<pa::Stream<pa::NonBlocking, pa::Input<f32>>>,
    rb_consumer: AudioConsumer,
    buffer: [f32; FRAMES],
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>
}

impl AudioBackend {
    fn new() -> AudioBackend {
        // todo, check result error...
        let pa = pa::PortAudio::new().unwrap();
        let def_input = pa.default_input_device().unwrap();
        let input_info = pa.device_info(def_input).unwrap();
        
        // create input device.__rust_force_expr!
        let latency = input_info.default_low_input_latency;
        let input_params = pa::StreamParameters::<f32>::new(def_input, CHANNELS, INTERLEAVED, latency);
    
        let settings = pa::InputStreamSettings::new(input_params,SAMPLE_RATE, FRAMES as u32);

        // We'll use this channel to send the count_down to the main thread for fun.
        //let (msg_sender, msg_receiver) = ::std::sync::mpsc::channel::<T>();
        
        let rb = ringbuf::HeapRb::<f32>::new(RINGBUF_SIZE);
        let (mut rb_producer, rb_consumer) = rb.split();

        let callback = move |pa::InputStreamCallbackArgs {
            buffer,
            frames,
            flags,
            time,
            ..
            }| 
        {
            assert!(frames == FRAMES as usize);

            if !rb_producer.is_full() {
                rb_producer.push_slice(buffer);
            }

            return pa::Continue;
        };

        let stream = Some(pa.open_non_blocking_stream(settings, callback).unwrap());

        let mut fft_planner = FftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(FRAMES);

        let audio_backend = AudioBackend {
            pa: pa,
            stream: stream,
            rb_consumer: rb_consumer,
            buffer: [0.0 ; FRAMES],
            fft: fft,
            fft_buffer: vec![Complex{ re: 0.0, im: 0.0 }; FRAMES],
        };
        
        return audio_backend;
    }

    fn start(&mut self) {
        self.stream.as_mut().unwrap().start();
    }

    fn on_tick(&mut self) {       
               
        if !self.rb_consumer.is_empty() {

            let max_frames = self.rb_consumer.pop_slice(&mut self.buffer);

            // setup for fft.
            for i in 0..FRAMES {
                self.fft_buffer[i].re = self.buffer[i];
                self.fft_buffer[i].im = 0.0;
            }

            // windowing todo

            // process fft.
            self.fft.process(&mut self.fft_buffer);
        }

    }

}

struct App {
    audio_backend: AudioBackend,
    data1: Vec<(f64, f64)>,
    data2: Vec<(f64, f64)>,
    window: [f64; 2],
}

impl App {
    fn new(audio_backend: AudioBackend) -> App {
        let data1 = Vec::<(f64, f64)>::new();
        let data2 = Vec::<(f64, f64)>::new();
        let mut app = App {
            audio_backend: audio_backend,
            data1: data1,
            data2: data2,
            window: [0.0, FRAMES as f64],
        };

        app.audio_backend.start();

        return app;
    }

    fn on_tick(&mut self) {
        self.audio_backend.on_tick();

        let mut off = 0.0;
        self.data1.clear();
        for frame in self.audio_backend.buffer {
            let mut data = (off, frame as f64);
            
            self.data1.push(data);

            off += 1.0;
        }

        off= 0.0;
        self.data2.clear();
        for frame in self.audio_backend.fft_buffer.as_mut_slice() {
            let mag = 10.0 * f32::log10((frame.re * frame.re) + (frame.im * frame.im) + 1e-20);
            let data = (off, mag as f64);           
            self.data2.push(data);

            off += 1.0;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // setup audio backend
    let audio_backend = AudioBackend::new();

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, DisableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let tick_rate = Duration::from_micros(16667);
    let app = App::new(audio_backend);
    let res = run_app(&mut terminal, app, tick_rate);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    tick_rate: Duration,
) -> io::Result<()> {
    let mut last_tick = Instant::now();
    terminal.clear();
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    return Ok(());
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}

fn ui<B: Backend>(f: &mut Frame<B>, app: &App) {
    let size = f.size();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Ratio(1, 2),
                Constraint::Ratio(1, 2),
            ]
            .as_ref(),
        )
        .split(size);
    let x_labels = vec![
        Span::styled(
            format!("{}", app.window[0]),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("{}", (app.window[0] + app.window[1]) / 2.0)),
        Span::styled(
            format!("{}", app.window[1]),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];
    let datasets = vec![
        Dataset::default()
            .name("Raw Data")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan))
            .data(&app.data1),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    "Waveform",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(x_labels)
                .bounds(app.window),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(vec![
                    Span::styled("-1", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("0"),
                    Span::styled("1", Style::default().add_modifier(Modifier::BOLD)),
                ])
                .bounds([-1.0, 1.0]),
        );
    f.render_widget(chart, chunks[0]);

    let x_labels = vec![
        Span::styled(
            format!("{}", app.window[0]),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("{}", (app.window[0] + app.window[1] / 8.0) / 2.0)),
        Span::styled(
            format!("{}", app.window[1] / 8.0),
            Style::default().add_modifier(Modifier::BOLD),
        ),
    ];
    let datasets = vec![Dataset::default()
        .name("data")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Yellow))
        .graph_type(GraphType::Line)
        .data(&app.data2)];
        let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    "FFT",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(x_labels)
                .bounds([0.0, (FRAMES / 8) as f64]),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(vec![
                    Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("0"),
                    Span::styled("50", Style::default().add_modifier(Modifier::BOLD)),
                ])
                .bounds([0.0, 50.0]),
        );
   f.render_widget(chart, chunks[1]);
}